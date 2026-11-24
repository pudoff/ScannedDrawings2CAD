const $=id=>document.getElementById(id);
function node(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}

export function createCorrections({request,onApplied}) {
  let current=null,epoch=0,controller=null,timer=null,packet=null,proposal=null,images=[],editSerial=0,sourceReview=null;
  function reset(){epoch++;editSerial++;sourceReview=null;controller?.abort();clearTimeout(timer);timer=null;current=null;packet=null;proposal=null;
    for(const url of images)URL.revokeObjectURL(url);images=[];
    $('correction-target').replaceChildren();$('correction-before').textContent='';$('correction-result').replaceChildren();
    $('correction-prepare').disabled=true;$('correction-accept').disabled=true;$('correction-approval').hidden=true;
    $('correction-note').value='';$('correction-new').value='';$('correction-review-ref').value='';
    $('correction-key').value='';$('correction-branch').checked=$('correction-derived').checked=false;
    $('correction-state').textContent='Выберите обработанный лист.';
  }
  const base=()=>`/v1/jobs/${current.jobId}/versions/${current.version}/pages/${current.page}/corrections`;
  async function context(ctx){
    const key=`${ctx.key}:${ctx.page}`;if(current?.key===key)return;reset();if(!ctx.ready)return;
    current={...ctx,key};controller=new AbortController();const own=epoch;
    try{const value=await request(base(),{signal:controller.signal});if(own!==epoch)return;packet=value;
      for(const target of value.targets){const option=node('option',`${{scalar:'Размер',ocr_reading:'Чтение',title_field:'Штамп'}[target.kind]} · ${target.label}`);
        option.value=String($('correction-target').options.length);$('correction-target').append(option);}
      $('correction-state').textContent=value.geometry_reason+' '+value.reading_reason;
      $('correction-prepare').disabled=!value.targets.length;targetChanged();
      $('correction-key-label').hidden=!value.permissions.requires_accept_key;
      const pending=value.proposals.find(p=>['prepared','queued','running','applied'].includes(p.status));
      if(pending){proposal=pending;await showProposal(own);}
    }catch(error){if(own===epoch)$('correction-state').textContent=error.message;}
  }
  function invalidate(){editSerial++;proposal=null;sourceReview=null;clearTimeout(timer);$('correction-approval').hidden=true;$('correction-accept').disabled=true;
    $('correction-branch').checked=$('correction-derived').checked=false;$('correction-review-ref').value='';
    $('correction-result').replaceChildren();for(const url of images)URL.revokeObjectURL(url);images=[];}
  function targetChanged(){invalidate();if(!packet)return;const target=packet.targets[Number($('correction-target').value)];
    if(!target)return;$('correction-before').textContent=`Сейчас: ${target.value===null?'не задано':target.value}`;
    $('correction-new').value='';$('correction-limits').hidden=target.kind!=='scalar';
    $('correction-effect').textContent=target.kind==='scalar'?'Изменятся связанные LINE/CIRCLE и зависимые размеры. Нужна проверка решения.':
      'Изменится проверенное чтение или поле штампа в метаданных. Исходная краска скана сохраняется.';
  }
  async function showProposal(own){
    if(own!==epoch||!proposal)return;const p=proposal;
    for(const url of images)URL.revokeObjectURL(url);images=[];
    $('correction-result').replaceChildren();$('correction-approval').hidden=true;$('correction-accept').disabled=true;
    $('correction-state').textContent=({queued:'Исправление в очереди…',running:'Проверяем исправление…',prepared:'Предложение готово. Проверьте до/после и зависимые размеры.',
      rejected:'Предложение отклонено проверками.',failed:'Исправление не выполнено: '+(p.error||'ошибка проверки'),
      stale:'Исходная версия устарела. Подготовьте исправление заново.',applied:`Исправление применено: версия ${p.applied_version}.`})[p.status]||p.status;
    if(p.preview_summary){const summary=p.preview_summary;
      const line=node('p',`${summary.target_id}: ${summary.before??'не задано'} → ${summary.after}`);$('correction-result').append(line);
      for(const update of summary.dimension_updates||[])$('correction-result').append(node('p',`${update.id}: ${update.old_nominal} → ${update.new_nominal}`));
      if(summary.effect)$('correction-result').append(node('p',summary.effect));
      if(summary.reason)$('correction-result').append(node('p',summary.reason));
      if(summary.preview_kind==='view_local_graph_not_DXF'){
        $('correction-result').append(node('p','Предпросмотр LINE/CIRCLE выбранного вида; остальные типы здесь не показаны. Это ещё не принятый DXF. Площади и опоры доступны в отчёте предложения.'));
        const pair=node('div');pair.className='correction-pair';$('correction-result').append(pair);
        for(const name of ['before','after']){try{const blob=await request(base()+`/${p.id}/images/${name}`,{signal:controller.signal},true);
            if(own!==epoch)return;const url=URL.createObjectURL(blob);images.push(url);const image=node('img');image.src=url;
            image.alt=name==='before'?'Граф до исправления':'Предложение после исправления';pair.append(image);
          }catch(error){if(own===epoch)$('correction-state').textContent=error.message;}}
      }
      const details=node('details'),title=node('summary','Отчёт предложения'),pre=node('pre',JSON.stringify(summary,null,2));details.append(title,pre);$('correction-result').append(details);
    }
    if(p.status==='prepared'){
      $('correction-approval').hidden=false;$('correction-accept').disabled=!(packet.permissions.can_accept||packet.permissions.requires_accept_key);
      if($('correction-accept').disabled)$('correction-state').textContent+=' Принятие для этого подключения отключено.';
    }
    if(p.status==='applied'){
      const open=node('button',`Открыть версию ${p.applied_version}`);open.type='button';open.className='text-button';open.id='correction-open-version';
      open.addEventListener('click',()=>onApplied(p.applied_version));$('correction-result').append(open);
    }
    if(['queued','running'].includes(p.status))timer=setTimeout(async()=>{
      try{const next=await request(base()+`/${p.id}`,{signal:controller.signal});if(own!==epoch)return;proposal=next;await showProposal(own);}
      catch(error){if(own===epoch)$('correction-state').textContent=error.message;}
    },700);
  }
  $('correction-target').addEventListener('change',targetChanged);
  for(const id of ['correction-new','correction-note','correction-displacement','correction-radius'])$(id).addEventListener('input',invalidate);
  $('correction-form').addEventListener('submit',async event=>{
    event.preventDefault();if(!packet)return;const own=epoch,editOwn=++editSerial,target=packet.targets[Number($('correction-target').value)];if(!target)return;
    const value=target.kind==='scalar'?Number($('correction-new').value):$('correction-new').value;
    const edit={source_manifest_sha256:packet.source_manifest_sha256,kind:target.kind,target_id:target.id,
      view_id:target.view_id||null,expected_old_value:target.value,new_value:value,evidence_ref:$('correction-note').value,
      max_point_displacement_mm:Number($('correction-displacement').value),max_radius_change_mm:Number($('correction-radius').value)};
    if(sourceReview)edit.review_id=sourceReview;
    $('correction-prepare').disabled=true;
    try{const next=await request(base()+'/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(edit),signal:controller.signal});
      if(own!==epoch||editOwn!==editSerial)return;proposal=next;clearTimeout(timer);await showProposal(own);
    }catch(error){if(own===epoch)$('correction-state').textContent=error.message;}
    finally{if(own===epoch)$('correction-prepare').disabled=false;}
  });
  $('correction-accept').addEventListener('click',async()=>{
    if(!proposal?.preview_summary)return;const own=epoch;
    if(!$('correction-branch').checked||!$('correction-derived').checked||!$('correction-review-ref').value.trim()){
      $('correction-state').textContent='Проверьте предложение, отметьте оба подтверждения и укажите основание проверки.';return;}
    const approval={proposal_sha256:proposal.preview_summary.proposal_sha256,branch_reviewed:true,derived_dimensions_reviewed:true,
      review_ref:$('correction-review-ref').value};
    const headers={'Content-Type':'application/json'};if($('correction-key').value)headers['X-Correction-Accept-Key']=$('correction-key').value;
    $('correction-accept').disabled=true;
    try{const next=await request(base()+`/${proposal.id}/accept`,{method:'POST',headers,body:JSON.stringify(approval),signal:controller.signal});
      if(own!==epoch)return;proposal=next;$('correction-key').value='';await showProposal(own);
    }catch(error){if(own===epoch){$('correction-state').textContent=error.message;$('correction-accept').disabled=false;}}
  });
  function fromReview(review){
    if(!packet||review.page_index!==current.page)return false;
    const titles={drawing_id:'document_number',part_name:'name',material:'material',person:'authors'};
    const index=packet.targets.findIndex(t=>titles[review.field]?t.kind==='title_field'&&t.id===titles[review.field]:
      review.field==='dimension'&&t.id===review.target&&['scalar','ocr_reading'].includes(t.kind));
    if(index<0)return false;
    $('correction-panel').open=true;$('correction-target').value=String(index);targetChanged();
    $('correction-new').value=review.value;$('correction-note').value=review.note||`review:${review.id}`;sourceReview=review.id;
    $('correction-new').focus();return true;
  }
  reset();return {context,reset,fromReview};
}
