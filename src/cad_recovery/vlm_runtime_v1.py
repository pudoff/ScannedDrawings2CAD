"""Offline Qwen adapter with bounded vision/text tokens and review-only output."""
import time

from PIL import Image

from .dataset_layout_v1 import checked
from .vlm_adapter_v1 import POLICY, SYSTEM, RequestBudget, envelope, prompt, trigger_allowed, validate_case, require


def locked_generation(model, requested):
    """HF4.57 model defaults must not turn an explicit greedy request into sampling."""
    value, _ = model._prepare_generation_config(requested, use_model_defaults=False,
        do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
        max_new_tokens=POLICY['maximum_new_tokens'])
    require(value.do_sample is False and value.num_beams == 1
            and value.get_generation_mode().value == 'greedy_search', 'VLM generation is not greedy')
    return value


class LocalVLM:
    def __init__(self, root, assets, maximum_requests=POLICY['maximum_requests']):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, GenerationConfig
        self.root = root; self.budget = RequestBudget(maximum_requests); self.generated_calls = 0
        for record in assets['files']: checked(root / assets['directory'], record)
        folder = root / assets['directory']
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16 CUDA is not supported')
        torch.manual_seed(20261111); torch.cuda.manual_seed_all(20261111); torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.fp32_precision = 'ieee'; torch.backends.cudnn.conv.fp32_precision = 'ieee'
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False,
            min_pixels=POLICY['min_processor_pixels'], max_pixels=POLICY['max_processor_pixels'])
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(folder), dtype=torch.bfloat16,
            attn_implementation=POLICY['attention'], local_files_only=True, trust_remote_code=False).cuda().eval()
        self.generation = GenerationConfig(max_new_tokens=POLICY['maximum_new_tokens'], do_sample=False,
            num_beams=1, use_cache=True, pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=[151645, 151643], repetition_penalty=1.0)
        self.generation = locked_generation(self.model, self.generation)
        torch.cuda.synchronize(); self.load_seconds = time.perf_counter() - started

    def infer(self, case, trigger):
        import torch
        validate_case(case)
        if not trigger_allowed(case, trigger): return envelope(case, skip='not_triggered')
        if case['task'] == 'candidate' and not case['candidates']: return envelope(case, skip='no_existing_candidates')
        if not self.budget.claim(): return envelope(case, skip='request_budget_exhausted')
        with Image.open(checked(self.root, case['image'])) as image:
            require(list(image.size) == case['size_px'], 'Changed fragment dimensions')
            image = image.convert('RGB')
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': [
            {'type': 'image', 'image': image}, {'type': 'text', 'text': prompt(case)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors='pt',
            min_pixels=POLICY['min_processor_pixels'], max_pixels=POLICY['max_processor_pixels'])
        tokens = inputs['input_ids'].shape[-1]
        grid = inputs['image_grid_thw'].tolist()
        vision_tokens = sum(int(t * h * w // 4) for t, h, w in grid)
        if tokens > POLICY['maximum_input_tokens'] or vision_tokens > POLICY['maximum_image_tokens']:
            return envelope(case, skip='input_token_budget_exceeded')
        inputs = {k: v.to(device='cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
        torch.cuda.synchronize(); started = time.perf_counter()
        self.generated_calls += 1
        with torch.inference_mode():
            generated = self.model.generate(**inputs, generation_config=self.generation, use_model_defaults=False,
                do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                max_new_tokens=POLICY['maximum_new_tokens'], max_time=POLICY['request_seconds'])
        torch.cuda.synchronize(); seconds = time.perf_counter() - started
        ids = generated[0, tokens:].cpu().tolist()
        raw = self.processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        value = envelope(case, raw, skip='request_time_exceeded' if seconds > POLICY['request_seconds'] else None)
        value['runtime'] = {'seconds': seconds, 'input_tokens': tokens, 'image_tokens': vision_tokens,
            'image_grid_thw': grid, 'new_tokens': len(ids), 'token_ids': ids}
        if torch.cuda.max_memory_allocated()/1024**2 > POLICY['allocated_MiB'] or torch.cuda.max_memory_reserved()/1024**2 > POLICY['reserved_MiB']:
            raise RuntimeError('VLM CUDA memory budget exceeded')
        return value
