const labels = {complete:'проверки выполнены', incomplete:'проверено частично', unknown:'полнота не записана', not_applicable:'не применяется'};
const stages = {seed:'Начальный поиск', width:'Поиск по толщине', axis:'Поиск по направлению', eligibility:'Отбор кандидатов', localized:'Уточнение острий', context:'Проверка окружения', curved:'Поиск на кривых'};
const causes = {arc_path_budget:'проверены не все дуговые участки',path_count_budget:'проверены не все пути',component_budget:'проверены не все локальные компоненты',profile_sample_budget:'исчерпан объём проверки пикселей',missing_explicit_producer_flag:'стадия не записала полноту',producer_reported_incomplete:'стадия сообщила об ограничении'};
export function coverageText(value) {
  if (!value || value.schema_version !== 'search-completeness/1.0') return {status:'unknown',title:'Поиск наконечников: полнота не записана',lines:['Для этой версии нет отдельных сведений о выполненном поиске.']};
  const status=value.domains?.head_search?.status || 'unknown';
  const lines=[`Структура CV: ${labels[value.domains?.structural_cv?.status] || labels.unknown}.`, `Начальный поиск: ${labels[value.domains?.head_seed?.status] || labels.unknown}.`, `Дополнительный поиск: ${labels[value.domains?.head_additions?.status] || labels.unknown}.`];
  for (const [name,row] of Object.entries(value.stages || {})) {
    if (row.status !== 'incomplete' && row.status !== 'unknown') continue;
    const reasons=(row.reasons || []).map(r=>causes[r.split(':').slice(1).join(':')] || 'есть ограничение или отсутствует запись полноты');
    lines.push(`${stages[name] || 'Проверка'}: ${[...new Set(reasons)].join('; ')}.`);
  }
  lines.push('Это статус выполнения проверок, а не точность распознавания.');
  return {status,title:`Поиск наконечников: ${labels[status] || labels.unknown}`,lines};
}
export function renderSearchCoverage(document,value,runStatus) {
  let panel=document.getElementById('search-coverage');const parent=document.getElementById('quality-message');
  if (!parent) return;
  if (!panel) {panel=document.createElement('details');panel.id='search-coverage';panel.className='notice';panel.setAttribute('aria-live','polite');parent.after(panel);}
  panel.hidden=runStatus !== 'completed';if(panel.hidden)return;
  const model=coverageText(value);panel.dataset.coverage=model.status;panel.replaceChildren();
  const title=document.createElement('summary');title.textContent=model.title;panel.append(title);
  for(const line of model.lines){const p=document.createElement('p');p.textContent=line;panel.append(p);}
}
