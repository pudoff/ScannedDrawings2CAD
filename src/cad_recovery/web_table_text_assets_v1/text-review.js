const $=id=>document.getElementById(id);
const node=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};

export function createTextReview({request,base,onProposal,draw,canvas,reviewer}){
  let epoch=0,key='',packet=null,proposal=null,rows=[],offset=0,drawing=null,operator='';
  const choices=new Map(),associations=new Map(),host=$('text-review');
  function reset(){epoch++;key='';packet=proposal=null;rows=[];offset=0;drawing=null;operator='';choices.clear();associations.clear();host.replaceChildren();}
  function controls(){const b=$('text-choice-preview');if(b)b.disabled=!operator.trim()||!$('text-choice-reviewed')?.checked||(!choices.size&&!associations.size);}
  function render(){
    host.replaceChildren();if(!packet)return;
    if(!packet.enabled){host.append(node('p',packet.reason));return;}
    const heading=node('h4','Чтения и альтернативы'),note=node('p','Выбор подготовит новый просмотр. Исходная версия и обе гипотезы сохраняются.');
    note.className='text-review-note';host.append(heading,note);
    const list=node('div');list.className='text-choice-list';
    for(const row of rows){
      const card=node('article');card.className='text-choice-card';
      if(row.kind==='unassigned'){
        card.append(node('strong',row.raw),node('p','Текст вне распознанных ячеек. Нужна явная область на листе.'));
        const show=node('button','Показать на листе');show.type='button';show.onclick=()=>draw(row.box_px);card.append(show);
        const associate=node('button',associations.has(row.handle)?'Изменить область':'Указать область');associate.type='button';
        associate.onclick=()=>{drawing=row;draw(row.box_px);$('text-choice-state').textContent='Обведите эту строку целиком на исходном листе.';canvas.scrollIntoView({block:'nearest'});};card.append(associate);
        if(associations.has(row.handle)){
          card.append(node('p','Область указана. Выбор будет применён в новом просмотре.'));
          const clear=node('button','Убрать привязку');clear.type='button';clear.onclick=()=>{associations.delete(row.handle);render();};card.append(clear);
        }
      }else{
        const cellKey=`${row.region_id}/${row.cell_id}`;
        card.append(node('strong',row.action==='conflict_keep_baseline'?'Чтения различаются':row.action==='agreement'?'Чтения совпали':'Проверьте чтение'));
        const meta=node('small',`${row.region_id} · ${row.cell_id}`);card.append(meta);
        for(const origin of ['baseline','candidate']){
          const values=row[origin];card.append(node('p',`${origin==='baseline'?'Прежнее':'Новое'}: ${values.map(v=>v.raw).join(' | ')||'нет чтения'}`));
          if(!row.ownership_ambiguous&&values.length){
            const button=node('button',origin==='baseline'?'Выбрать прежнее':'Выбрать новое');button.type='button';
            const selected=choices.get(cellKey);button.setAttribute('aria-pressed',String(selected?.origin===origin));
            button.onclick=()=>{choices.set(cellKey,{region_id:row.region_id,cell_id:row.cell_id,origin,handles:values.map(v=>v.handle)});draw(values[0].box_px);render();};card.append(button);
            if(values.length>1)for(const item of values){
              const single=node('button',`Только «${item.raw}»`);single.type='button';single.className='text-single-choice';
              single.onclick=()=>{choices.set(cellKey,{region_id:row.region_id,cell_id:row.cell_id,origin,handles:[item.handle]});draw(item.box_px);render();};card.append(single);
            }
          }
        }
        if(row.ownership_ambiguous)card.append(node('p','Границы ячейки неоднозначны. Чтения сохранены для проверки.'));
        if(choices.has(cellKey)){
          const clear=node('button','Оставить без ручного выбора');clear.type='button';clear.onclick=()=>{choices.delete(cellKey);render();};card.append(clear);
        }
      }
      list.append(card);
    }
    host.append(list);
    if(packet.read_only){
      for(const button of list.querySelectorAll('button'))if(button.textContent!=='Показать на листе')button.disabled=true;
      if(packet.has_more){const more=node('button','Показать ещё');more.type='button';more.onclick=()=>loadMore();host.append(more);}
      if(operator)host.append(node('p',`Проверил выбор: ${operator}`));
      host.append(node('p','Выбор сохранён в этой версии. Прежние чтения и области доступны для проверки.'));
      return;
    }
    if(packet.has_more){const more=node('button','Показать ещё');more.type='button';more.onclick=()=>loadMore();host.append(more);}
    const status=node('p',`Выбрано ячеек: ${choices.size}. Привязано строк: ${associations.size}.`);status.id='text-choice-state';status.setAttribute('role','status');host.append(status);
    const person=node('label','Проверил выбор');const name=node('input');name.id='text-choice-reviewer';name.maxLength=120;name.value=operator;
    name.placeholder='Имя или идентификатор';name.oninput=()=>{operator=name.value;controls();};person.append(name);host.append(person);
    const label=node('label');label.className='text-choice-confirm';const checked=node('input');checked.type='checkbox';checked.id='text-choice-reviewed';checked.onchange=controls;
    label.append(checked,document.createTextNode('Проверены выбранные чтения и области'));host.append(label);
    const prepare=node('button','Подготовить выбранный текст');prepare.type='button';prepare.id='text-choice-preview';prepare.onclick=submit;host.append(prepare);controls();
  }
  async function loadMore(){const own=epoch;try{const suffix=packet.read_only?'/published/text':`/${proposal.id}/text`;const value=await request(base()+`${suffix}?offset=${offset}&limit=40`);if(own!==epoch)return;packet=value;rows.push(...value.rows);offset+=value.rows.length;render();}catch(e){if(own===epoch)host.append(node('p',e.message));}}
  async function show(row,current){
    const next=`${current.jobId}:${current.version}:${current.page}:${row.id}:${row.preview_sha256||row.status}`;
    if(['queued','running','failed','cancelled'].includes(row.status)){
      if(next!==key){reset();key=next;proposal=row;}
      const own=epoch;
      try{const value=await request(base()+`/${row.id}/diagnostics`);if(own===epoch)host.replaceChildren(node('p',[value.reason,value.phase,Number.isFinite(value.elapsed_seconds)?`${Math.round(value.elapsed_seconds)} с`:null].filter(Boolean).join(' · ')||'Ожидаем исполнителя.'));}catch{if(own===epoch)host.replaceChildren(node('p','Диагностика временно недоступна.'));}
      return;
    }
    if(next===key)return;reset();key=next;proposal=row;
    if(!['prepared','applied'].includes(row.status))return;
    const own=epoch;
    try{const value=await request(base()+`/${row.id}/text?offset=0&limit=40`);if(own!==epoch)return;
      packet=value;rows=value.rows||[];offset=rows.length;
      operator=value.previous_reviewer||'';
      for(const choice of value.previous_choice?.selections||[])choices.set(`${choice.region_id}/${choice.cell_id}`,choice);
      for(const association of value.previous_choice?.associations||[])associations.set(association.handle,association);
      render();
    }catch(e){if(own===epoch)host.append(node('p',e.message));}
  }
  async function showPublished(value){
    reset();packet=value;rows=value.rows||[];offset=rows.length;operator=value.previous_reviewer||'';
    for(const choice of value.previous_choice?.selections||[])choices.set(`${choice.region_id}/${choice.cell_id}`,choice);
    for(const association of value.previous_choice?.associations||[])associations.set(association.handle,association);
    render();
  }
  function associate(box){
    if(!drawing)return;const row=drawing;drawing=null;
    if(box[0]<=row.box_px[0]&&box[1]<=row.box_px[1]&&box[2]>=row.box_px[2]&&box[3]>=row.box_px[3]){
      associations.set(row.handle,{handle:row.handle,box_px:box});render();
    }else $('text-choice-state').textContent='Область должна содержать строку целиком. Укажите её снова.';
  }
  async function submit(){
    const own=epoch;const b=$('text-choice-preview');b.disabled=true;
    try{
      const row=await request(base()+`/${proposal.id}/text-preview`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({proposal_id:proposal.id,preview_sha256:proposal.preview_sha256,review_sha256:packet.review_sha256,
          reviewer:operator.trim(),reviewed:true,selections:[...choices.values()],associations:[...associations.values()]})});
      if(own===epoch)await onProposal(row);
    }catch(e){if(own===epoch){$('text-choice-state').textContent=e.message;controls();}}
  }
  return {show,showPublished,reset,isDrawing:()=>!!drawing,associate};
}

export function createRuntimeStatus({request}){
  let epoch=0,current=null,timer=null;const host=$('runtime-status');
  function reset(){epoch++;clearTimeout(timer);current=null;host.textContent='';}
  async function poll(own){
    if(!current?.jobId)return;
    try{const value=await request(`/v1/jobs/${current.jobId}/versions/${current.version}/diagnostics`);
      if(own!==epoch)return;
      host.textContent=value.available?[value.reason,value.phase,
        Number.isFinite(value.elapsed_seconds)?`${Math.round(value.elapsed_seconds)} с`:null,
        Number.isFinite(value.peak_memory_MiB)?`Память: ~${Math.round(value.peak_memory_MiB)} МиБ`:null].filter(Boolean).join(' · '):'Подробная диагностика этой версии недоступна.';
    }catch{if(own===epoch)host.textContent='Диагностика временно недоступна.';}
    if(own===epoch&&['queued','running'].includes(current.status))timer=setTimeout(()=>poll(own),1500);
  }
  function context(c){if(!c.jobId){reset();return;}if(current?.key===c.key&&current?.status===c.status)return;reset();current=c;poll(epoch);}
  return {context,reset};
}
