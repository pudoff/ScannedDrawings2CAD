import {createInspector} from './inspector.js';
import {createCorrections} from './corrections.js';
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = { token: '', online: false, file: null, uploading: false, jobId: null, version: null, job: null,
    bundle: null, bundleKey: '', page: 0, mode: 'source', zoom: 1, previewKey: '', previewUrl: null,
    documents: [], nextOffset: null, query: '', listEpoch: 0, jobEpoch: 0, requestEpoch: 0, refreshing: false, saving: false, creating: false, optionsKey: '', reviewContext: '' };
  const statusLabels = { queued: 'В очереди', running: 'Обрабатывается', completed: 'Обработано', failed: 'Не удалось обработать' };
  const fieldLabels = { comment: 'Замечание', drawing_id: 'Номер чертежа', part_name: 'Наименование', material: 'Материал', person: 'ФИО', dimension: 'Размер' };
  const errorLabels = { input_invalid: 'Содержимое или размер листа не поддерживается. Проверьте файл и ограничения во вкладке «Как это работает».',
    input_integrity: 'Исходный файл изменился. Загрузите оригинал заново.', processing_timeout: 'Обработка заняла слишком много времени. Попробуйте отдельный лист или меньший файл.',
    output_limit: 'Результат превысил допустимый объём. Попробуйте отдельный лист.', attempts_exhausted: 'Обработку не удалось завершить после перезапусков worker.',
    processing_failed: 'Не удалось прочитать или обработать файл. Проверьте исходник или загрузите его заново.' };
  let toastTimer, searchTimer;
  function node(tag, className, text) { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; }
  function date(value) { return value ? new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value)) : '—'; }
  function size(bytes) { return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} МБ` : `${Math.max(1, Math.round(bytes / 1024))} КБ`; }
  function context() { return `${state.jobId}:${state.version}:${state.jobEpoch}`; }
  function run() { return state.job?.versions.find(v => v.version === state.version); }
  function toast(text, error = false) { clearTimeout(toastTimer); $('toast').textContent = text; $('toast').classList.toggle('error', error); $('toast').hidden = false; toastTimer = setTimeout(() => { $('toast').hidden = true; }, 5500); }
  function authorize() { if (!$('auth-dialog').open) $('auth-dialog').showModal(); }
  function failure(status) { return ({ 400: 'Файл или его содержимое не прошли проверку.', 401: 'Введите ключ доступа к сервису.', 404: 'Документ или файл не найден.',
    409: 'Результат ещё не готов. Дождитесь обработки.', 413: 'Превышен допустимый размер данных.', 415: 'Поддерживаются PDF, PNG и JPEG.',
    422: 'Проверьте заполненные поля и параметры.', 503: 'Сервис временно недоступен. Попробуйте ещё раз.' })[status] || 'Не удалось выполнить действие. Попробуйте ещё раз.'; }
  function sameOrigin(path) { const url = new URL(path, location.origin); if (url.origin !== location.origin) throw new Error('Недопустимый адрес сервиса.'); return url; }
  async function request(path, options = {}, asBlob = false) {
    let response;
    try { response = await fetch(sameOrigin(path), { ...options, headers: { ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}), ...options.headers } }); }
    catch { throw new Error('Нет связи с сервисом. Проверьте подключение.'); }
    if (!response.ok) { let message=failure(response.status);if(path.includes('/corrections')){try{const value=await response.json();if(typeof value.detail==='string')message=value.detail.slice(0,600);}catch{}}
      const error = new Error(message); error.status = response.status; if (response.status === 401) authorize(); throw error; }
    return asBlob ? response.blob() : response.json();
  }
  function connected(ok) {
    state.online = ok; $('connection-dot').className = `status-dot ${ok ? 'online' : 'offline'}`;
    $('connection-label').textContent = ok ? 'Сервис доступен' : 'Нет подключения'; $('offline-banner').hidden = ok; uploadControls();
  }
  async function health() { try { await request('/health'); connected(true); return true; } catch (e) { connected(false); if (e.status === 401) $('connection-label').textContent = 'Нужен ключ'; return false; } }
  function uploadControls() {
    $('submit-upload').disabled = !state.file || !state.online || state.uploading;
    $('submit-upload').firstElementChild.textContent = state.uploading ? 'Загружаем файл…' : 'Начать обработку';
    $('file-input').disabled = state.uploading; $('clear-file').disabled = state.uploading;
  }
  function chooseFile(file) {
    if (!file) return;
    if (!/\.(pdf|png|jpe?g)$/i.test(file.name)) { toast('Выберите PDF, PNG или JPEG.', true); return; }
    if (file.size > 50 * 1024 * 1024 || !file.size) { toast('Файл должен быть непустым и не больше 50 МБ.', true); return; }
    state.file = file; $('chosen-name').textContent = file.name; $('chosen-size').textContent = size(file.size);
    $('chosen-file').hidden = false; $('dropzone').hidden = true; uploadControls();
  }
  function clearFile() { state.file = null; $('file-input').value = ''; $('chosen-file').hidden = true; $('dropzone').hidden = false; uploadControls(); }
  function options() {
    const value = { profile: $('profile').value, render_dpi: Number($('render-dpi').value), image_dpi: $('image-dpi').value ? Number($('image-dpi').value) : null, rotation_deg: Number($('rotation').value) };
    if (!Number.isFinite(value.render_dpi) || value.render_dpi < 36 || value.render_dpi > 300 || (value.image_dpi !== null && (value.image_dpi < 36 || value.image_dpi > 1200))) throw new Error('Проверьте разрешение листа в параметрах.');
    return value;
  }
  async function upload(event) {
    event.preventDefault(); if (!state.file || state.uploading) return;
    let settings; try { settings = options(); } catch (e) { toast(e.message, true); return; }
    const file = state.file; state.uploading = true; uploadControls(); $('upload-progress').hidden = false; $('upload-progress').value = 0;
    try {
      const parameters = new URLSearchParams({ filename: file.name }); for (const [key, value] of Object.entries(settings)) if (value !== null) parameters.set(key, value);
      const result = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest(); xhr.open('POST', `/v1/jobs?${parameters}`); xhr.setRequestHeader('Content-Type', file.type || 'application/octet-stream');
        if (state.token) xhr.setRequestHeader('Authorization', `Bearer ${state.token}`);
        xhr.upload.onprogress = e => { if (e.lengthComputable) $('upload-progress').value = 100 * e.loaded / e.total; };
        xhr.onload = () => { if (xhr.status === 202) { try { resolve(JSON.parse(xhr.responseText)); } catch { reject(new Error('Не удалось прочитать ответ сервиса.')); } }
          else { if (xhr.status === 401) authorize(); reject(new Error(failure(xhr.status))); } };
        xhr.onerror = () => reject(new Error('Загрузка прервалась. Проверьте подключение.')); xhr.timeout = 120000; xhr.ontimeout = () => reject(new Error('Истекло время загрузки файла.')); xhr.send(file);
      });
      clearFile(); await loadDocuments(); await selectJob(result.job_id, result.version); toast('Файл загружен и поставлен в очередь.');
      workspaceView('viewer');
    } catch (e) { toast(e.message, true); }
    finally { state.uploading = false; $('upload-progress').hidden = true; uploadControls(); }
  }
  async function loadDocuments(append = false) {
    const epoch = ++state.listEpoch; const offset = append ? state.nextOffset : 0;
    if (offset === null) return;
    try {
      const data = await request(`/v1/jobs?${new URLSearchParams({ limit: 30, offset, query: state.query })}`);
      if (epoch !== state.listEpoch) return;
      state.documents = append ? state.documents.concat(data.documents) : data.documents; state.nextOffset = data.next_offset;
      $('document-count').textContent = data.total; $('more-documents').hidden = data.next_offset === null; renderDocuments();
    } catch (e) { if (epoch === state.listEpoch) { if (!state.documents.length) $('documents-list').replaceChildren(node('p', 'list-empty', e.message)); } }
  }
  function renderDocuments() {
    const list = $('documents-list'); list.replaceChildren();
    if (!state.documents.length) { list.append(node('p', 'list-empty', state.query ? 'Ничего не найдено.' : 'Документов пока нет.\nНачните с загрузки чертежа.')); return; }
    for (const d of state.documents) {
      const button = node('button', `job-item${d.id === state.jobId ? ' selected' : ''}`); button.type = 'button'; button.setAttribute('aria-pressed', String(d.id === state.jobId));
      button.append(node('span', 'status-dot')); const copy = node('div'); const title = node('strong', '', d.original_name); title.title = d.original_name;
      copy.append(title, node('small', '', `${statusLabels[d.status] || d.status} · v${d.latest_version} · ${date(d.created_at)}`)); button.append(copy);
      button.addEventListener('click', () => selectJob(d.id)); list.append(button);
    }
  }
  function clearPreview(resetInspector = true) {
    if (resetInspector) {inspector.reset();corrections.reset();}
    state.previewKey = ''; if (state.previewUrl) URL.revokeObjectURL(state.previewUrl); state.previewUrl = null;
    $('preview-image').hidden = true; $('preview-image').removeAttribute('src'); $('empty-preview').hidden = false; $('preview-loading').hidden = true;
    state.zoom = 1; applyZoom();
  }
  async function selectJob(id, version = null) {
    tab('preview');
    state.jobEpoch++; state.jobId = id; state.version = version; state.job = null; state.bundle = null; state.bundleKey = ''; state.page = 0; state.mode = 'dxf'; clearPreview();
    $('document-title').textContent = 'Загружаем документ…'; $('version-select').disabled = true;
    renderDocuments(); renderState(); await refreshJob();
  }
  async function refreshJob() {
    const id = state.jobId; if (!id) return; const epoch = state.jobEpoch; const requestEpoch = ++state.requestEpoch;
    try {
      const job = await request(`/v1/jobs/${encodeURIComponent(id)}`); if (state.jobId !== id || epoch !== state.jobEpoch || requestEpoch !== state.requestEpoch) return;
      const previous = run()?.status; state.job = job;
      if (!job.versions.some(v => v.version === state.version)) state.version = job.latest_version;
      const selected = run(); history.replaceState(null, '', `#job=${id}&v=${state.version}`);
      if (state.optionsKey !== context() && !state.file) {
        const o = selected.options; $('profile').value = o.profile; $('render-dpi').value = o.render_dpi; $('image-dpi').value = o.image_dpi ?? ''; $('rotation').value = o.rotation_deg; state.optionsKey = context();
      }
      $('document-title').textContent = job.original_name; $('document-title').title = job.original_name;
      const select = $('version-select'); select.replaceChildren(); for (const v of job.versions) { const option = node('option', '', `v${v.version} · ${statusLabels[v.status]}`); option.value = v.version; select.append(option); } select.value = state.version;
      select.disabled = false; $('refresh-job').disabled = false;
      if (selected?.status !== previous) $('announcement').textContent = `${job.original_name}: ${statusLabels[selected.status]}`;
      renderState(); renderHistory();
      if (selected.status === 'completed' && state.bundleKey !== context()) await loadCompleted();
      if (selected.status !== 'completed') { state.bundle = null; state.bundleKey = ''; clearPreview(); renderIssues(); }
    } catch (e) { if (state.jobId === id && epoch === state.jobEpoch && requestEpoch === state.requestEpoch) toast(e.message, true); }
  }
  async function loadCompleted() {
    const key = context(); const path = `/v1/jobs/${state.jobId}/versions/${state.version}`;
    try {
      const [bundle, notes] = await Promise.all([request(path + '/artifacts'), request(path + '/reviews')]);
      if (context() !== key) return;
      state.bundle = bundle; state.bundleKey = key; state.page = Math.min(state.page, Math.max(0, bundle.quality.page_count - 1));
      renderState(); renderIssues(); renderReviews(notes.reviews); renderArtifacts(); renderPage();
    } catch (e) { if (context() === key) toast(e.message, true); }
  }
  function renderState() {
    const r = run(); const status = r?.status; const result = $('result-status'); result.dataset.state = status || '';
    $('empty-preview').querySelector('h4').textContent = ({ queued: 'Чертёж в очереди', running: 'Обрабатываем чертёж', failed: 'Обработка не завершена' })[status] || 'Здесь появится чертёж';
    $('empty-preview').querySelector('p').textContent = ['queued', 'running'].includes(status) ? 'Лист появится после завершения обработки. Статус обновляется автоматически.' : status === 'failed' ? 'Посмотрите причину в блоке результата.' : 'Загрузите файл или выберите документ из списка слева.';
    result.lastElementChild.textContent = statusLabels[status] || 'Ожидаем ваш файл';
    $('result-description').textContent = ({ queued: 'Файл в очереди. Можно продолжать работу с другими документами.', running: 'Готовим результат. Статус обновится автоматически.',
      completed: `${r?.summary?.page_count || 0} лист(ов) · версия ${r?.version}. Результат доступен для просмотра.`, failed: errorLabels[r?.error?.code] || 'Не удалось завершить обработку.' })[status] || 'Выберите чертёж, чтобы начать работу.';
    document.querySelectorAll('#processing-steps li').forEach((item, i) => { item.className = ''; if (!status) return; const step = ({ queued: 0, running: 1, completed: 3, failed: 1 })[status]; if (i < step || (status === 'queued' && i === 0)) item.classList.add('done'); if (i === step && status !== 'failed') item.classList.add('current'); });
    const warning = $('quality-message'); warning.hidden = !['completed', 'failed'].includes(status); warning.classList.toggle('error', status === 'failed');
    warning.textContent = status === 'failed' ? 'Исходник сохранён. Можно создать новую версию или загрузить другой файл.' : r?.summary?.status === 'needs_review' ? 'Нужна проверка. Размеры, масштаб и геометрия ещё не подтверждены.' : 'Обработка завершена. Проверьте результат перед использованием.';
    const ready = status === 'completed' && state.bundleKey === context(); $('review-fields').disabled = !ready || state.saving;
    if (ready) showExportStatus();
    $('reprocess-button').disabled = $('reprocess-history').disabled = !['completed', 'failed'].includes(status) || state.creating;
    $('download-dxf').disabled = !ready || !pageArtifact('dxf') || pageExport()?.empty === true; $('artifact-details').hidden = !ready;
    for (const mode of ['source', 'dxf', 'geometry', 'ocr']) $(`mode-${mode}`).disabled = !ready || !previewArtifact(mode);
    $('mode-comparison').disabled = !ready;
    ['zoom-in', 'zoom-out', 'zoom-fit'].forEach(id => { $(id).disabled = !ready || !previewArtifact(state.mode); });
    if (!ready) { $('previous-page').disabled = $('next-page').disabled = true; $('page-label').textContent = 'Лист —'; $('reviews-list').replaceChildren(); $('artifact-list').replaceChildren(); }
  }
  function pageArtifact(extension) { const prefix = `page-${String(state.page + 1).padStart(3, '0')}`; return state.bundle?.artifacts.find(a => a.filename === `${prefix}.${extension}`); }
  function pageExport() { return state.bundle?.quality.pages?.find(p => p.page_index === state.page)?.demo_export; }
  function showExportStatus() {
    if (run()?.options.profile === 'native_observations') { $('quality-message').hidden = false; $('quality-message').textContent = 'Native-наблюдения требуют проверки. DXF этим режимом не создаётся. Стыки, конфликты и неполнота поиска перечислены в замечаниях; JSON доступен в других файлах.'; return; }
    const report = pageExport(); if (!report) return;
    const warning = $('quality-message'); warning.hidden = false; warning.classList.toggle('error', report.empty === true);
    warning.textContent = report.empty ? 'Геометрия не восстановлена: этот DXF пуст. Диагностические файлы доступны ниже.' :
      `${report.mode === 'raster_ink_draft' ? 'Штриховой черновик скана' : 'Векторный черновик PDF'} · ${report.geometry_entity_count} объектов. Размеры и масштаб не подтверждены.`;
    if (report.input_resized) {
      const original = report.original_size_px; const analysis = report.analysis_size_px;
      warning.textContent += Array.isArray(original) && original.length === 2 && Array.isArray(analysis) && analysis.length === 2 ?
        ` Для показа уменьшено с ${original[0]}×${original[1]} до ${analysis[0]}×${analysis[1]} px; оригинал сохранён.` :
        ' Для показа использована уменьшенная рабочая копия; оригинал сохранён.';
    }
  }
  function previewArtifact(mode) {
    const prefix = `page-${String(state.page + 1).padStart(3, '0')}`;
    const candidates = state.bundle?.artifacts || [];
    if (mode === 'source') return candidates.find(a => a.filename === `${prefix}-source.png`);
    if (mode === 'dxf') return candidates.find(a => a.filename === `${prefix}-dxf.png`);
    if (mode === 'ocr') return candidates.find(a => a.filename === `${prefix}-ocr-preview.png`);
    return candidates.find(a => a.filename.startsWith(prefix) && a.filename.endsWith('.png') && /geometry|candidates/.test(a.filename));
  }
  const inspector = createInspector({request, onReview: id => { tab('reviews'); $('review-target').value = id; $('review-note').focus(); }});
  const corrections = createCorrections({request,onApplied:async version=>{await changeVersion(version);await loadDocuments();tab('observations');}});
  function renderPage() {
    if (state.reviewContext !== `${context()}:${state.page}`) { $('review-field').value = 'comment'; $('review-value-wrap').hidden = true; $('review-target').value = $('review-value').value = $('review-note').value = ''; state.reviewContext = `${context()}:${state.page}`; }
    inspector.context({jobId:state.jobId, version:state.version, page:state.page, key:context(), ready:!!state.bundle});
    corrections.context({jobId:state.jobId, version:state.version, page:state.page, key:context(), ready:!!state.bundle});
    const total = state.bundle?.quality.page_count || 0; $('page-label').textContent = total ? `Лист ${state.page + 1} из ${total}` : 'Лист —';
    $('previous-page').disabled = state.page <= 0; $('next-page').disabled = state.page >= total - 1;
    $('review-page').textContent = `Правка к листу ${state.page + 1} · версия ${state.version}`;
    if (!previewArtifact(state.mode)) state.mode = 'source';
    for (const mode of ['source', 'dxf', 'geometry', 'ocr']) { const button = $(`mode-${mode}`); button.classList.toggle('active', state.mode === mode); button.setAttribute('aria-pressed', String(state.mode === mode)); button.disabled = !previewArtifact(mode); }
    $('preview-caption').textContent = ({source: run()?.options.profile === 'native_observations' ? 'Навигационный предпросмотр · EXIF/поворот учтены · анализ без уменьшения' : pageExport()?.source_preview_resized ? 'Рабочая копия исходника · уменьшена для показа' : 'Исходный лист · без изменений', dxf: 'Отрисовка скачиваемого DXF · черновик', geometry: 'Кандидаты геометрии · нужна проверка', ocr: 'OCR-наблюдения · нужна проверка'})[state.mode];
    $('download-dxf').disabled = !pageArtifact('dxf') || pageExport()?.empty === true; showExportStatus(); state.zoom = 1; applyZoom(); loadPreview();
  }
  async function loadPreview() {
    const artifact = previewArtifact(state.mode); const key = `${context()}:${state.page}:${state.mode}`; state.previewKey = key;
    if (!artifact) { clearPreview(false); return; }
    $('preview-image').hidden = true; $('preview-loading').hidden = false;
    try {
      const blob = await request(artifact.download_url, {}, true); const url = URL.createObjectURL(blob);
      if (state.previewKey !== key) { URL.revokeObjectURL(url); return; }
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl); state.previewUrl = url;
      $('preview-image').src = url; $('preview-image').alt = `${({source: pageExport()?.source_preview_resized ? 'Уменьшенная рабочая копия исходника' : 'Исходный лист', dxf: 'Отрисовка DXF', geometry: 'Кандидаты геометрии', ocr: 'OCR-наблюдения'})[state.mode]}, лист ${state.page + 1}, ${state.job.original_name}`;
      $('preview-image').hidden = false; $('empty-preview').hidden = true;
    } catch (e) { if (state.previewKey === key) toast(e.message, true); }
    finally { if (state.previewKey === key) $('preview-loading').hidden = true; }
  }
  function applyZoom() { $('preview-stage').classList.toggle('zoomed', state.zoom !== 1); $('preview-stage').style.setProperty('--zoom', state.zoom); $('zoom-fit').textContent = state.zoom === 1 ? 'По размеру' : `${Math.round(state.zoom * 100)}%`; }
  function renderIssues() {
    const items = state.bundle?.quality.pages.flatMap(p => (p.issues || []).map(i => ({ ...i, page: p.page_index + 1 }))) || [];
    $('issue-count').textContent = items.length; const list = $('issues-list'); list.replaceChildren();
    if (!items.length) { list.append(node('p', 'list-empty', state.bundle ? 'В отчёте нет замечаний. Проверьте результат перед использованием.' : 'Замечания появятся после обработки.')); return; }
    for (const issue of items) { const row = node('article', 'issue-item'); row.append(node('span', 'issue-marker', '↗')); const text = node('div'); text.append(node('h4', '', `Лист ${issue.page} · ${issue.severity === 'error' ? 'Требует внимания' : 'Нужна проверка'}`), node('p', '', issue.message)); row.append(text); list.append(row); }
  }
  function renderHistory() {
    const list = $('history-list'); list.replaceChildren();
    for (const v of [...(state.job?.versions || [])].reverse()) { const row = node('article', 'history-item'); const button = node('button'); button.type = 'button'; button.append(node('strong', '', `Версия ${v.version}`), node('span', '', v.version === state.version ? 'Выбрана' : 'Открыть ↗')); button.addEventListener('click', () => changeVersion(v.version));
      row.append(button, node('p', '', `${statusLabels[v.status]} · ${date(v.created_at)}${v.options._manual_correction?' · ручное исправление из v'+v.options._manual_correction.parent_version:''}`), node('small', '', ({native_observations: 'Native-анализ · без DXF', presentation_draft: 'DXF для просмотра + анализ', geometry_observations: 'Диагностика геометрии', baseline: 'Извлечение векторов PDF'})[v.options.profile] || v.options.profile)); list.append(row); }
  }
  async function changeVersion(version) { state.jobEpoch++; state.version = Number(version); state.bundle = null; state.bundleKey = ''; state.page = 0; state.mode = 'dxf'; clearPreview(); renderState(); await refreshJob(); }
  function renderArtifacts() {
    const list = $('artifact-list'); list.replaceChildren(); $('artifact-count').textContent = state.bundle?.artifacts.length || 0;
    for (const a of state.bundle?.artifacts || []) { const button = node('button', 'artifact-item'); button.type = 'button'; button.title = a.filename; button.append(node('span', '', a.filename), node('span', '', `${size(a.size_bytes)} ↓`)); button.addEventListener('click', () => download(a)); list.append(button); }
  }
  async function download(artifact) {
    if (!artifact) return;
    try { const blob = await request(artifact.download_url, {}, true); const url = URL.createObjectURL(blob); const link = node('a'); link.href = url; link.download = artifact.filename; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000); toast(`Скачивание: ${artifact.filename}`); }
    catch (e) { toast(e.message, true); }
  }
  function renderReviews(reviews) {
    const list = $('reviews-list'); list.replaceChildren();
    for (const review of [...reviews].reverse()) { const item = node('article', 'review-item'); item.append(node('strong', '', `${fieldLabels[review.field] || review.field}${review.page_index !== null ? ` · лист ${review.page_index + 1}` : ''}`));
      if (review.target) item.append(node('p', '', review.target)); if (review.value) item.append(node('p', '', review.value)); if (review.note) item.append(node('p', '', review.note)); item.append(node('small', '', date(review.created_at)));
      if(review.value){const apply=node('button','text-button','Подготовить применение');apply.type='button';apply.addEventListener('click',()=>{if(!corrections.fromReview(review))toast('Выберите тот же лист и точный поддерживаемый ID; размерная геометрия требует проверенного графа.',true);});item.append(apply);}list.append(item); }
  }
  async function saveReview(event) {
    event.preventDefault(); if (state.saving || !state.bundle) return; const key = `${context()}:${state.page}`; const path = `/v1/jobs/${state.jobId}/versions/${state.version}/reviews`;
    const data = { field: $('review-field').value, page_index: state.page, target: $('review-target').value, value: $('review-value').value, note: $('review-note').value };
    if (data.field === 'comment' && !data.note.trim()) { toast('Добавьте комментарий.', true); $('review-note').focus(); return; }
    if (data.field !== 'comment' && !data.value.trim()) { toast('Укажите предложенное значение.', true); $('review-value').focus(); return; }
    state.saving = true; renderState();
    try { await request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
      if (`${context()}:${state.page}` === key) { $('review-target').value = $('review-value').value = $('review-note').value = ''; const result = await request(path); if (`${context()}:${state.page}` === key) renderReviews(result.reviews); }
      toast('Правка сохранена отдельно от автоматической версии.');
    } catch (e) { toast(e.message, true); } finally { state.saving = false; renderState(); }
  }
  async function reprocess() {
    if (!state.jobId || state.creating) return; const id = state.jobId; let settings; try { settings = options(); } catch (e) { toast(e.message, true); return; }
    state.creating = true; renderState();
    try { const result = await request(`/v1/jobs/${id}/versions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings) });
      await loadDocuments(); if (state.jobId === id) await changeVersion(result.version); toast(`Создана версия ${result.version}. Предыдущий результат сохранён.`);
    } catch (e) { toast(e.message, true); } finally { state.creating = false; renderState(); }
  }
  function workspaceView(name) {
    $('workbench').dataset.activePanel = name;
    for (const button of document.querySelectorAll('[data-workspace-view]')) button.setAttribute('aria-pressed', String(button.dataset.workspaceView === name));
  }
  for (const button of document.querySelectorAll('[data-workspace-view]')) button.addEventListener('click', () => workspaceView(button.dataset.workspaceView));
  function tab(name, focus = false) {
    workspaceView('viewer');
    for (const button of document.querySelectorAll('[data-tab]')) { const active = button.dataset.tab === name; button.classList.toggle('active', active); button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1; $(`view-${button.dataset.tab}`).hidden = !active; if (active && focus) button.focus(); }
  }
  $('file-input').addEventListener('change', e => chooseFile(e.target.files[0])); $('clear-file').addEventListener('click', clearFile); $('upload-form').addEventListener('submit', upload);
  ['dragenter', 'dragover'].forEach(type => $('dropzone').addEventListener(type, e => { e.preventDefault(); $('dropzone').classList.add('drag-over'); }));
  ['dragleave', 'drop'].forEach(type => $('dropzone').addEventListener(type, e => { e.preventDefault(); $('dropzone').classList.remove('drag-over'); if (type === 'drop') chooseFile(e.dataTransfer.files[0]); }));
  document.addEventListener('dragover', e => e.preventDefault()); document.addEventListener('drop', e => e.preventDefault());
  $('document-search').addEventListener('input', e => { clearTimeout(searchTimer); state.query = e.target.value; searchTimer = setTimeout(() => loadDocuments(), 250); });
  $('more-documents').addEventListener('click', () => loadDocuments(true)); $('version-select').addEventListener('change', e => changeVersion(e.target.value));
  $('previous-page').addEventListener('click', () => { if (state.page > 0) { state.page--; renderPage(); } }); $('next-page').addEventListener('click', () => { if (state.page + 1 < (state.bundle?.quality.page_count || 0)) { state.page++; renderPage(); } });
  $('mode-comparison').addEventListener('click', () => tab('observations'));
  $('mode-source').addEventListener('click', () => { state.mode = 'source'; renderPage(); }); $('mode-geometry').addEventListener('click', () => { state.mode = 'geometry'; renderPage(); });
  $('mode-dxf').addEventListener('click', () => { state.mode = 'dxf'; renderPage(); }); $('mode-ocr').addEventListener('click', () => { state.mode = 'ocr'; renderPage(); });
  $('zoom-in').addEventListener('click', () => { state.zoom = Math.min(3, state.zoom + .25); applyZoom(); }); $('zoom-out').addEventListener('click', () => { state.zoom = Math.max(.5, state.zoom - .25); applyZoom(); }); $('zoom-fit').addEventListener('click', () => { state.zoom = 1; applyZoom(); });
  $('download-dxf').addEventListener('click', () => download(pageArtifact('dxf'))); $('reprocess-button').addEventListener('click', reprocess); $('reprocess-history').addEventListener('click', reprocess);
  $('review-field').addEventListener('change', e => { $('review-value-wrap').hidden = e.target.value === 'comment'; }); $('review-form').addEventListener('submit', saveReview);
  $('help-button').addEventListener('click', () => $('help-dialog').showModal()); document.querySelectorAll('[data-close-dialog]').forEach(b => b.addEventListener('click', () => b.closest('dialog').close()));
  $('focus-documents').addEventListener('click', () => { tab('documents'); $('document-search').focus(); });
  document.querySelectorAll('[data-tab]').forEach(b => { b.addEventListener('click', () => tab(b.dataset.tab)); b.addEventListener('keydown', e => { const names = [...document.querySelectorAll('[data-tab]')].map(t => t.dataset.tab); let index = names.indexOf(b.dataset.tab); if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); index = (index + (e.key === 'ArrowRight' ? 1 : names.length-1)) % names.length; tab(names[index], true); } }); });
  $('auth-form').addEventListener('submit', async e => { e.preventDefault(); state.token = $('auth-token').value; $('auth-error').hidden = true;
    if (await health()) { $('auth-dialog').close(); $('auth-token').value = ''; await bootstrap(); } else $('auth-error').hidden = false; });
  $('refresh-job').addEventListener('click', async () => { if (await health()) { inspector.reset(); state.bundleKey = ''; await refreshJob(); await loadDocuments(); } });
  async function reconnect() { if (await health()) { await loadDocuments(); await refreshJob(); } }
  $('retry-connection').addEventListener('click', reconnect); $('connection-button').addEventListener('click', reconnect);
  async function bootstrap() {
    if (!await health()) return; await loadDocuments(); const hash = new URLSearchParams(location.hash.slice(1)); const id = hash.get('job');
    if (id && /^[0-9a-f-]{36}$/i.test(id)) await selectJob(id, Number(hash.get('v')) || null);
    tab('preview');
    if (!state.jobId) workspaceView('upload');
  }
  setInterval(async () => { if (document.hidden || !state.online || state.refreshing || !state.jobId) return; state.refreshing = true;
    try { if (['queued', 'running'].includes(run()?.status)) { await refreshJob(); await loadDocuments(); } } finally { state.refreshing = false; } }, 2000);
  setInterval(() => { if (!document.hidden && !$('auth-dialog').open) health(); }, 20000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) reconnect(); });
  bootstrap();
})();
