"""Versioned late-response guards over the immutable V27 UI sources.

No API, worker, processing profile or CAD/OCR changes. Each hook is counted so
a changed upstream asset fails closed instead of silently losing a guard.
"""


def replace_once(source, old, new):
    if source.count(old)!=1:
        raise ValueError('UI response-scope hook differs: '+old[:90])
    return source.replace(old,new,1)


def app(source):
    source=replace_once(source,'  let toastTimer, searchTimer;','  let toastTimer, searchTimer, completedSerial=0, previewSerial=0;')
    source=replace_once(source,'    const key = context(); const path =',
        '    const own=++completedSerial; const key = context(); const path =')
    source=replace_once(source,'      if (context() !== key) return;',
        '      if (context() !== key || own!==completedSerial) return;')
    source=replace_once(source,'if (context() === key) toast(e.message, true);',
        'if (context() === key && own===completedSerial) toast(e.message, true);')
    source=replace_once(source,'    const artifact = previewArtifact(state.mode); const key =',
        '    const own=++previewSerial; const artifact = previewArtifact(state.mode); const key =')
    source=replace_once(source,'if (state.previewKey !== key) {',
        'if (state.previewKey !== key || own!==previewSerial) {')
    source=source.replace('if (state.previewKey === key)', 'if (state.previewKey === key && own===previewSerial)')
    source=replace_once(source,'    if (!artifact) return;\n    try { const blob =',
        '    if (!artifact) return; const selection=context(), page=state.page;\n    try { const blob =')
    source=replace_once(source,'const blob = await request(artifact.download_url, {}, true); const url = URL.createObjectURL(blob); const link =',
        'const blob = await request(artifact.download_url, {}, true); if(selection!==context()||page!==state.page)return; const url = URL.createObjectURL(blob); const link =')
    source=replace_once(source,'    if (!state.jobId || state.creating) return; const id = state.jobId;',
        '    if (!state.jobId || state.creating) return; const selectionEpoch=state.jobEpoch; const id = state.jobId;')
    source=replace_once(source,'if (state.jobId === id) await changeVersion(result.version);',
        'if (state.jobId === id && state.jobEpoch===selectionEpoch) await changeVersion(result.version);')
    source=replace_once(source,'    const file = state.file; state.uploading = true;',
        '    const selectionEpoch=state.jobEpoch; const file = state.file; state.uploading = true;')
    source=replace_once(source,'clearFile(); await loadDocuments(); await selectJob(result.job_id, result.version);',
        'clearFile(); await loadDocuments(); if(state.jobEpoch===selectionEpoch)await selectJob(result.job_id, result.version);')
    source=replace_once(source,"clearTimeout(searchTimer); state.query = e.target.value;",
        "clearTimeout(searchTimer); state.listEpoch++; state.query = e.target.value;")
    return source


def corrections(source):
    begin=source.index('  async function showProposal(own){');end=source.index("  $('correction-target').addEventListener",begin)
    section=source[begin:end]
    section=replace_once(section,'const p=proposal;',
        'const p=proposal, edit=editSerial; const alive=()=>own===epoch&&edit===editSerial&&proposal?.id===p.id;')
    section=section.replace('if(own!==epoch)return;','if(!alive())return;')
    section=section.replace('if(own===epoch)', 'if(alive())')
    section=replace_once(section,'      const details=', '      if(!alive())return;\n      const details=')
    section=replace_once(section,"    if(p.status==='prepared'){", "    if(!alive())return;\n    if(p.status==='prepared'){")
    section=replace_once(section,'timer=setTimeout(async()=>{','timer=setTimeout(async()=>{\n      if(!alive())return;')
    source=source[:begin]+section+source[end:]
    source=replace_once(source,"if(!proposal?.preview_summary)return;const own=epoch;",
        "if(!proposal?.preview_summary)return;const own=epoch, edit=editSerial;")
    source=replace_once(source,"if(own!==epoch)return;proposal=next;$('correction-key')",
        "if(own!==epoch||edit!==editSerial)return;proposal=next;$('correction-key')")
    source=replace_once(source,"catch(error){if(own===epoch){$('correction-state').textContent=error.message;$('correction-accept')",
        "catch(error){if(own===epoch&&edit===editSerial){$('correction-state').textContent=error.message;$('correction-accept')")
    source=replace_once(source,"}catch(error){if(own===epoch)$('correction-state').textContent=error.message;}\n    finally{if(own===epoch)$('correction-prepare')",
        "}catch(error){if(own===epoch&&editOwn===editSerial)$('correction-state').textContent=error.message;}\n    finally{if(own===epoch&&editOwn===editSerial)$('correction-prepare')")
    source=replace_once(source,"  function invalidate(){editSerial++;proposal=null;",
        "  function invalidate(){editSerial++;$('correction-prepare').disabled=!packet?.targets.length;proposal=null;")
    return source


def tables(source):
    source=replace_once(source,"        const links=node('div');", "        if(own!==epoch||revision!==serial)return;\n        const links=node('div');")
    source=replace_once(source,'    await textReview.show(row,current);',
        '    await textReview.show(row,current);if(own!==epoch||revision!==serial)return;')
    source=replace_once(source,'  async function download(kind){\n    try{',
        '  async function download(kind){\n    const own=epoch,revision=serial,page=current.page;\n    try{')
    source=replace_once(source,'{},true);const url=URL.createObjectURL(blob);',
        '{},true);if(own!==epoch||revision!==serial)return;const url=URL.createObjectURL(blob);')
    source=replace_once(source,'`tables-page-${current.page+1}.dxf`','`tables-page-${page+1}.dxf`')
    source=replace_once(source,"if(!proposal||!$('table-reviewed').checked)return;const own=epoch;",
        "if(!proposal||!$('table-reviewed').checked)return;const own=epoch,revision=serial;")
    source=replace_once(source,'});if(own!==epoch)return;await onApplied(row.version);',
        '});if(own!==epoch||revision!==serial)return;await onApplied(row.version);')
    source=replace_once(source,"}catch(e){$('table-state').textContent=e.message;}\n  }\n  async function show(",
        "}catch(e){if(own===epoch&&revision===serial)$('table-state').textContent=e.message;}\n  }\n  async function show(")
    source=replace_once(source,"}catch(e){if(own===epoch)$('table-state').textContent=e.message;}\n        if(own!==epoch||revision!==serial)return;",
        "}catch(e){if(own===epoch&&revision===serial)$('table-state').textContent=e.message;}\n        if(own!==epoch||revision!==serial)return;")
    source=replace_once(source,"}catch(e){if(own===epoch){$('table-state').textContent=e.message;controls();}}",
        "}catch(e){if(own===epoch&&edit===serial){$('table-state').textContent=e.message;controls();}}")
    return source


def text_review(source):
    source=replace_once(source,"  let epoch=0,key=''", "  let loadingMore=false,diagnosticsSerial=0;\n  let epoch=0,key=''")
    source=replace_once(source,"  function reset(){epoch++;key='';", "  function reset(){epoch++;loadingMore=false;diagnosticsSerial++;key='';")
    source=replace_once(source,'  async function loadMore(){const own=epoch;try{',
        '  async function loadMore(){if(loadingMore)return;loadingMore=true;const own=epoch;try{')
    source=replace_once(source,"catch(e){if(own===epoch)host.append(node('p',e.message));}}\n  async function show(",
        "catch(e){if(own===epoch)host.append(node('p',e.message));}finally{if(own===epoch)loadingMore=false;}}\n  async function show(")
    source=replace_once(source,'      const own=epoch;\n      try{const value=await request(base()+`/${row.id}/diagnostics`);',
        '      const own=epoch,revision=++diagnosticsSerial;\n      try{const value=await request(base()+`/${row.id}/diagnostics`);')
    source=replace_once(source,"if(own===epoch)host.replaceChildren(node('p',[value.reason", "if(own===epoch&&revision===diagnosticsSerial)host.replaceChildren(node('p',[value.reason")
    source=replace_once(source,"catch{if(own===epoch)host.replaceChildren(node('p','Диагностика временно недоступна.'));}",
        "catch{if(own===epoch&&revision===diagnosticsSerial)host.replaceChildren(node('p','Диагностика временно недоступна.'));}")
    return source


def inspector(source):
    return replace_once(source,"}catch(error){if(epoch===serial)$('comparison-state').textContent=error.message;}\n  });",
        "}catch(error){if(epoch===serial&&own===maskEpoch)$('comparison-state').textContent=error.message;}\n  });")


def upgrade(path,source):
    function={'app.js':app,'corrections.js':corrections,'tables.js':tables,'text-review.js':text_review,'inspector.js':inspector}.get(path.rsplit('/',1)[-1])
    return function(source) if function else source
