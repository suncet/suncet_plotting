import { GROUPS, SIGNALS, POWER_PAIRS } from './signals.mjs';
import { loadDatabase, loadSnapshot, numericValue, stateLabel, sampleSeries, powerValue } from './data.mjs';
import { selectRange, panelGroups, derivePowerSignal } from './plot-policy.mjs';

const $ = id => document.getElementById(id);
const colors = ['#ffab69','#75d9c1','#81b6ff','#d6a2f5','#f7d275','#ed8199','#65c4da','#abb5fa','#bedc85','#dcab88','#c3d5e7','#f38dd7'];
const icons = {overview:'◫',temperatures:'♨',electrical:'ϟ',modes:'◈',attitude:'⌖',timing:'◷',heaters:'≋',storage:'▤',science:'✧'};
let dataset, entries = [], activeGroup = 'overview', plots = [], zoom = null, loading = false, syncing = false, renderVersion = 0, mode='public', archive=[], loadedCaptureId;
let plotlyPromise;
const number = value => Number(value).toLocaleString('en-US');
const fmt = value => value === null || value === undefined ? '—' : Math.abs(value) >= 10000 ? Number(value).toPrecision(4) : Number(value).toLocaleString('en-US',{maximumFractionDigits:3});
const available = signal => dataset?.columns.includes(signal.field);
const signalsByField = new Map(SIGNALS.map(signal=>[signal.field,signal]));
const matches = signal => `${signal.label} ${signal.field} ${signal.unit} ${signal.note || ''}`.toLowerCase().includes($('search').value.trim().toLowerCase());
const pairMatches = pair => `${pair.label} ${pair.voltage} ${pair.current} power voltage current`.toLowerCase().includes($('search').value.trim().toLowerCase());
const value = (row, signal) => {
  const raw=signal.derive?signal.derive(row):row[signal.field];
  if(raw===null || raw===undefined || raw==='')return null;
  return signal.kind === 'state' ? stateLabel(signal,raw) : numericValue(raw);
};
const axisTitle = () => ({record:'Beacon sample · acquisition order',boot:'Time since boot (s)',header:'CCSDS onboard seconds · coarse + milliseconds'})[$('axis').value];

function element(tag, className, text) {
  const result = document.createElement(tag);
  if(className) result.className = className;
  if(text !== undefined) result.textContent = text;
  return result;
}
function message(text, error=false) {
  $('message').textContent = text;
  $('message').classList.toggle('error',error);
  $('message').hidden = !text;
}
function getPlotly() {
  if (!plotlyPromise) plotlyPromise = new Promise((resolve,reject) => {
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-3.1.0.min.js';
    script.onload = () => resolve(window.Plotly);
    script.onerror = () => { plotlyPromise = null; script.remove(); reject(new Error('Could not load the plot library. Check your internet connection and try opening the file again.')); };
    document.head.append(script);
  });
  return plotlyPromise;
}

function navigation() {
  $('navigation').replaceChildren();
  const groups = [{id:'overview',label:'Overview'},...GROUPS];
  groups.forEach(group => {
    const button = element('button',activeGroup===group.id?'active':'');
    button.append(element('span','nav-icon',icons[group.id] || '◇'),element('span','',group.label));
    if(dataset && group.id !== 'overview') button.append(element('span','count',SIGNALS.filter(s=>s.group===group.id && available(s)).length));
    button.setAttribute('aria-current',activeGroup===group.id?'page':'false');
    button.onclick = () => { activeGroup=group.id; navigation(); render().catch(showError); };
    $('navigation').append(button);
  });
}

function getEntries() {
  const reset = $('reset-filter').value, axis=$('axis').value;
  const result = entries.filter(e => (reset==='all'||String(e.row.beac_num_sc_resets)===reset) && (!$('duplicates').checked || !e.row.is_duplicate_packet))
    .map(e => {
      let x=e.index;
      if(axis==='boot') x=numericValue(e.row.beac_time_since_boot);
      if(axis==='header') {
        const sec=numericValue(e.row.ccsdsSecHeader2_sec_beacon), sub=numericValue(e.row.ccsdsSecHeader2_sub_beacon);
        // A bad fine field is missing time, never silently zero milliseconds.
        x=sec !== null && sec>=0 && sub !== null && Number.isInteger(sub) && sub>=0 && sub<1000 ? sec+sub/1000 : null;
      }
      return {...e,x};
    }).filter(e=>e.x!==null);
  if(axis!=='record') result.sort((a,b)=>a.x-b.x || a.index-b.index);
  return result;
}
function selectedSignals(group) { return SIGNALS.filter(s => available(s) && (!group || s.group===group) && matches(s)); }
function section(title, detail='') {
  const head=element('div','section-heading');head.append(element('h2','',title),element('span','',detail));$('plots').append(head);
  const grid=element('div','chart-grid');$('plots').append(grid);return grid;
}
function panel(grid,title,subtitle,unit='',wide=false) {
  const container=element('article',`panel${wide?' wide':''}`);
  const head=element('div','panel-header'), names=element('div');
  names.append(element('h3','panel-title',title),element('div','panel-subtitle',subtitle));head.append(names);
  if(unit) head.append(element('span','panel-unit',unit));container.append(head);
  const chart=element('div','chart');chart.setAttribute('aria-label',`${title} interactive chart`);container.append(chart);grid.append(container);
  return {container,chart};
}
function baseLayout(height=310) {
  return {height,autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{family:'DM Sans, system-ui, sans-serif',color:'#94a5bc',size:10},
    margin:{l:55,r:25,t:18,b:70},colorway:colors,dragmode:'zoom',hovermode:'x unified',
    xaxis:{title:{text:axisTitle(),font:{size:9},standoff:12},gridcolor:'#283440',zeroline:false,showline:true,linecolor:'#344150',range:zoom || undefined,autorange:!zoom},
    yaxis:{gridcolor:'#283440',zeroline:false,tickfont:{size:9},automargin:true},
    legend:{orientation:'h',y:-.28,x:0,font:{size:9},bgcolor:'transparent'},hoverlabel:{bgcolor:'#233245',font:{size:11,color:'#e9f1fa'}},uirevision:`${activeGroup}-${$('axis').value}-${$('reset-filter').value}`};
}
async function plot(chart,traces,layout,version) {
  if(version!==renderVersion) return;
  await window.Plotly.newPlot(chart,traces,layout,{responsive:true,displaylogo:false,scrollZoom:false,modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d','resetScale2d'],toImageButtonOptions:{format:'png',scale:2}});
  if(version!==renderVersion){window.Plotly.purge(chart);return;}
  plots.push(chart);
  chart.on('plotly_relayout',event=> {
    if(syncing) return;
    if(event['xaxis.autorange']) setZoom(null);
    else if(event['xaxis.range[0]']!==undefined && event['xaxis.range[1]']!==undefined) setZoom([Number(event['xaxis.range[0]']),Number(event['xaxis.range[1]'])]);
    else if(event['xaxis.range']) setZoom(event['xaxis.range'].map(Number));
  });
  chart.on('plotly_doubleclick',()=>{setZoom(null);return false;});
}
async function setZoom(range) {
  zoom=range;syncing=true;
  $('range-note').textContent=range ? `Selected: ${fmt(range[0])} → ${fmt(range[1])} · CSV uses this range` : 'Drag a plot to zoom every panel.';
  // Resample the underlying records after every zoom, so fine structure returns.
  try { await render(); }
  finally {syncing=false;}
}

function applyDisplayRange(axis,signals,values,fitValues=values) {
  const selected=$('y-scale').value;
  const result=selectRange(signals,selected==='engineering'?fitValues:values,selected);
  if(result.range){axis.range=result.range;axis.autorange=false;}
  // Power fitting checks its source channels, but the count describes raw watts.
  if(result.range)result.outside=values.map(numericValue).filter(v=>v!==null&&(v<result.range[0]||v>result.range[1])).length;
  return result;
}
function rangeCaption(result,unit='') {
  if(!result.range)return '';
  const range=`${fmt(result.range[0])}–${fmt(result.range[1])}${unit?` ${unit}`:''}`;
  return `Display ${range}${result.outside?` · ${number(result.outside)} values outside view`:''}`;
}

async function numericPanel(grid,title,signals,view,version,{wide=false,subtitle='',height=330}={}) {
  if(!signals.length || version!==renderVersion) return;
  const units=[...new Set(signals.map(s=>s.unit).filter(Boolean))];
  const single=signals.length===1;
  const p=panel(grid,title,subtitle || (single?'Latest sample in selected interval':`${signals.length} channels · shared interval`),units.join(' / '),wide);
  if(single){
    const signal=signals[0];const latest=view.filter(e=>!zoom||e.x>=zoom[0]&&e.x<=zoom[1]).at(-1);
    const lastValue=latest?numericValue(signal.derive?signal.derive(latest.row):latest.row[signal.field]):null;
    const badge=p.container.querySelector('.panel-unit');
    if(badge){badge.textContent=`${fmt(lastValue)} ${signal.unit || ''}`;badge.classList.add('panel-reading');badge.title='Last sample in the selected interval, before display limits';}
    p.container.querySelector('.panel-title').title=signal.field || signal.label;
  }
  const xs=view.map(e=>e.x);
  const traces=signals.map((signal,i)=>{
    const series=sampleSeries(xs,view.map(e=>numericValue(signal.derive?signal.derive(e.row):e.row[signal.field])),1800,'number');
    return {x:series.x,y:series.y,type:'scatter',mode:'lines',name:signal.label,line:{color:colors[i%colors.length],width:1.5,shape:signal.step?'hv':'linear'},connectgaps:false,hovertemplate:`%{y:.5g} ${signal.unit || ''}<extra>%{fullData.name}</extra>`};
  });
  const layout=baseLayout(height);layout.yaxis.title={text:units.join(' / '),font:{size:9}};
  const rawValues=signals.flatMap(s=>view.map(e=>s.derive?s.derive(e.row):e.row[s.field]));
  const fitValues=signals.flatMap(s=>view.map(e=>s.rangeValue?s.rangeValue(e.row):s.derive?s.derive(e.row):e.row[s.field]));
  const range=applyDisplayRange(layout.yaxis,signals,rawValues,fitValues);
  if(range.range)p.container.append(element('div','panel-footer',rangeCaption(range,units.join(' / '))));
  layout.showlegend=!single;
  layout.margin.b=single?50:signals.length>6?110:80;layout.legend.y=signals.length>6?-.29:-.35;
  if(signals.length>6) layout.height=height+50;
  p.chart.style.height=`${layout.height}px`;
  await plot(p.chart,traces,layout,version);
}

function stateColor(state,signal) {
  if(signal.field.startsWith('beac_eps_pwr_state_')){
    if(state==='ON')return '#36b879';
    if(state==='OFF')return '#ee6262';
  }
  let hash=0;for(const c of String(state)) hash=(hash*31+c.charCodeAt(0))>>>0;
  return colors[hash%colors.length];
}
async function statePanel(grid,title,signals,view,version) {
  if(!signals.length || !view.length || version!==renderVersion) return;
  const p=panel(grid,title,'Named states · hover for the decoded value · gaps mean missing data','STATE',true);
  const byState=new Map();
  const deltas=view.slice(1).map((e,i)=>e.x-view[i].x).filter(d=>d>0).sort((a,b)=>a-b);
  const endStep=deltas[Math.floor(deltas.length/2)] || 1;
  signals.forEach(signal=>{
    let start=view[0].x, previous=value(view[0].row,signal);
    const add=(end)=>{
      if(previous===null || previous===undefined || end<=start) return;
      const state=String(previous);
      const color=stateColor(state,signal), key=`${state}\0${color}`;
      if(!byState.has(key)) byState.set(key,{x:[],base:[],y:[],text:[],name:state,type:'bar',orientation:'h',textposition:'none',marker:{color},hovertemplate:'%{y}: <b>%{text}</b><extra></extra>'});
      const trace=byState.get(key);trace.x.push(end-start);trace.base.push(start);trace.y.push(signal.label);trace.text.push(state);
    };
    for(let i=1;i<view.length;i++) {
      const next=value(view[i].row,signal);
      if(next!==previous){add(view[i].x);start=view[i].x;previous=next;}
    }
    add(view.at(-1).x+endStep);
  });
  const layout=baseLayout(Math.max(280,signals.length*31+150));
  layout.barmode='overlay';layout.bargap=.32;layout.hovermode='closest';layout.margin={l:155,r:25,t:15,b:100};
  layout.yaxis={type:'category',autorange:'reversed',categoryorder:'array',categoryarray:signals.map(s=>s.label),tickfont:{size:10},automargin:true};
  layout.legend.y=-.22;layout.xaxis.type='linear';p.chart.style.height=`${layout.height}px`;
  await plot(p.chart,[...byState.values()],layout,version);
}

async function electrical(view,version) {
  const grid=section('Paired electrical measurements','VOLTAGE / CURRENT / DERIVED POWER');grid.className='electrical-grid';
  const pairs=POWER_PAIRS.filter(pair=>dataset.columns.includes(pair.voltage)||dataset.columns.includes(pair.current))
    .filter(pairMatches);
  for(const pair of pairs){
    if(version!==renderVersion)return;
    const both=dataset.columns.includes(pair.voltage)&&dataset.columns.includes(pair.current);
    const p=panel(grid,pair.label,pair.note || 'Matching voltage and current from the same beacon packet','V / A / W');
    const power=row=>powerValue(row,pair);
    const definitions=[{...signalsByField.get(pair.voltage),field:pair.voltage,label:'Voltage',unit:'V',color:colors[2]},{...signalsByField.get(pair.current),field:pair.current,label:'Current',unit:'A',color:colors[1]},{...derivePowerSignal(pair),label:pair.note?.toLowerCase().includes('charge')?'Charge power':'Power',derive:power,color:colors[0]}];
    const readings=element('div','rail-readings');
    const last=view.filter(e=>!zoom||e.x>=zoom[0]&&e.x<=zoom[1]).at(-1);
    definitions.forEach(d=>{const item=element('div','rail-reading');item.style.color=d.color;item.append(element('small','',d.label),document.createTextNode(fmt(last?(d.derive?d.derive(last.row):numericValue(last.row[d.field])):null)),element('span','',` ${d.unit}`));readings.append(item);});
    p.container.insertBefore(readings,p.chart);
    const xs=view.map(e=>e.x);
    const traces=definitions.map((d,i)=>{const s=sampleSeries(xs,view.map(e=>d.derive?d.derive(e.row):numericValue(e.row[d.field])),1400);return {x:s.x,y:s.y,name:d.label,type:'scatter',mode:'lines',line:{color:d.color,width:1.5},yaxis:i===0?'y':`y${i+1}`,connectgaps:false,hovertemplate:`%{y:.5g} ${d.unit}<extra>%{fullData.name}</extra>`};});
    const layout=baseLayout(370);layout.showlegend=false;layout.margin.b=50;
    layout.yaxis={...layout.yaxis,domain:[.73,1],title:{text:'V',font:{color:colors[2],size:10}}};
    layout.yaxis2={...layout.yaxis,domain:[.37,.64],title:{text:'A',font:{color:colors[1],size:10}}};
    layout.yaxis3={...layout.yaxis,domain:[0,.27],title:{text:'W',font:{color:colors[0],size:10}}};layout.xaxis.anchor='y3';
    const ranges=definitions.map((d,i)=>applyDisplayRange(layout[i===0?'yaxis':`yaxis${i+1}`],[d],view.map(e=>d.derive?d.derive(e.row):e.row[d.field]),view.map(e=>d.rangeValue?d.rangeValue(e.row):d.derive?d.derive(e.row):e.row[d.field])));
    const outside=ranges.reduce((sum,result)=>sum+result.outside,0);
    p.chart.style.height='370px';p.container.append(element('div','panel-footer',both?'Values above: last packet in selection · P = V × I':'A required channel is missing; derived power is unavailable.'));
    if(ranges.some(result=>result.range))p.container.append(element('div','panel-footer',`${$('y-scale').selectedOptions[0].textContent}${outside?` · ${number(outside)} values outside view`:''}`));
    await plot(p.chart,traces,layout,version);
  }
  const paired=new Set(POWER_PAIRS.flatMap(p=>[p.voltage,p.current]));
  const extras=selectedSignals('electrical').filter(s=>!paired.has(s.field));
  for(const signal of extras) await numericPanel(grid,signal.label,[signal],view,version,{height:260,subtitle:signal.note || 'Independent voltage reference'});
  if(!pairs.length&&!extras.length)grid.append(element('div','empty','No electrical telemetry points match this search.'));
}

async function render() {
  const group=GROUPS.find(g=>g.id===activeGroup);
  $('page-title').textContent=group?.label || 'Beacon overview';
  $('page-description').textContent=group?.description || (dataset?'An organized view of spacecraft health, power, and operating states.':'Open a capture. Explore the spacecraft.');
  if(!dataset) return;
  const version=++renderVersion;
  plots.forEach(p=>window.Plotly.purge(p));plots=[];$('plots').replaceChildren();$('plots').setAttribute('aria-busy','true');
  const fullView=getEntries();
  let view=fullView;
  if(zoom){
    const start=fullView.findIndex(e=>e.x>=zoom[0]);
    const stop=fullView.findIndex(e=>e.x>zoom[1]);
    // Include neighbors to retain line/step continuity at the view boundary.
    view=start<0?fullView.slice(-1):fullView.slice(Math.max(0,start-1),stop<0?undefined:stop+1);
  }
  const exactCount=zoom?fullView.filter(e=>e.x>=zoom[0]&&e.x<=zoom[1]).length:view.length;
  const resetCount=new Set(view.map(e=>e.row.beac_num_sc_resets).filter(v=>v!==null&&v!==undefined)).size;
  const latest=view.filter(e=>!zoom||e.x>=zoom[0]&&e.x<=zoom[1]).at(-1);
  const resetNumber=latest?numericValue(latest.row.beac_num_sc_resets):null;
  const signalCount=SIGNALS.filter(available).length;
  const powerCount=POWER_PAIRS.filter(p=>dataset.columns.includes(p.voltage)&&dataset.columns.includes(p.current)).length;
  $('metrics').replaceChildren();
  [['BEACON SAMPLES',number(exactCount),`of ${number(dataset.rows.length)} loaded records`],['TELEMETRY POINTS',signalCount,'Available in this capture'],['RESET NUMBER',resetNumber===null?'—':number(resetNumber),`Latest sample · ${resetCount} distinct resets in view`],['DERIVED POWER CHANNELS',powerCount,'Matching V × I, in watts']].forEach(([label,count,note])=>{
    const card=element('div','metric');card.append(element('div','label',label),element('strong','',count),element('small','',note));$('metrics').append(card);
  });
  const axis=$('axis').value;
  $('time-note').textContent=axis==='record'?'Packet order keeps resets and clock jumps visible. Select a reset and time axis for time-based plots.':`${axis==='boot'?'Boot time may restart':'Onboard time is shown as raw seconds, without assuming UTC'}.${resetCount>1?' Multiple resets selected: equal times can overlap.':''} Rows with missing or invalid time are omitted.`;
  if(!view.length){$('plots').append(element('div','empty','No samples have a valid value for this selection and horizontal axis.'));$('plots').setAttribute('aria-busy','false');return;}
  const search=$('search').value.trim();
  if(activeGroup==='electrical') await electrical(view,version);
  else if(activeGroup==='overview'&&!search){
    let grid=section('Thermal & electrical trends','ALL CHANNELS, SHARED INTERVAL');
    await numericPanel(grid,'Spacecraft temperatures',selectedSignals('temperatures'),view,version,{wide:true,subtitle:'Solar arrays, batteries, avionics, instruments, ADCS, and radios'});
    if(version!==renderVersion)return;
    const powerSignals=POWER_PAIRS.filter(p=>dataset.columns.includes(p.voltage)&&dataset.columns.includes(p.current)).map(p=>({...derivePowerSignal(p),label:p.label,derive:row=>powerValue(row,p)}));
    await numericPanel(grid,'Electrical power',powerSignals,view,version,{wide:true,subtitle:'Measured voltage × matching current · batteries show charge power'});
    if(version!==renderVersion)return;
    grid=section('Operating modes','DECODED STATES');
    await statePanel(grid,'Spacecraft & subsystem modes',selectedSignals('modes').filter(s=>['beac_mode_system_mode','beac_adcs_mode','beac_csie_cap_state','beac_eps_pwr_state_adcs','beac_eps_pwr_state_csie','beac_eps_pwr_state_dsps','beac_eps_pwr_state_uhf','beac_eps_pwr_state_xband'].includes(s.field)),view,version);
  }else{
    const signals=selectedSignals(activeGroup==='overview'?null:activeGroup);
    if(!signals.length) $('plots').append(element('div','empty','No matching telemetry points in this database. Try another group or search.'));
    const grid=section(search?`Telemetry points matching “${search}”`:group?.label || 'Beacon telemetry points',`${signals.length} TELEMETRY POINTS`);
    const nums=signals.filter(s=>s.kind!=='state').map(s=>({...s,step:['timing','storage'].includes(s.group)}));
    if(activeGroup==='overview'){
      for(const unit of [...new Set(nums.map(s=>s.unit))]){
        const groupSignals=nums.filter(s=>s.unit===unit);
        await numericPanel(grid,unit?`Measurements · ${unit}`:'Measurements',groupSignals,view,version,{wide:groupSignals.length>4});
      }
    }else{
      grid.classList.add('signal-grid');
      for(const group of panelGroups(nums))await numericPanel(grid,group.title,group.signals,view,version,{height:260});
    }
    const states=signals.filter(s=>s.kind==='state');
    if(activeGroup==='modes'){
      const powerStates=states.filter(s=>s.field.startsWith('beac_eps_pwr_state_'));
      const otherStates=states.filter(s=>!s.field.startsWith('beac_eps_pwr_state_'));
      await statePanel(grid,'Operating modes & system states',otherStates,view,version);
      await statePanel(grid,'Subsystem power states',powerStates,view,version);
    }else await statePanel(grid,'States & transitions',states,view,version);
  }
  if(version!==renderVersion)return;
  // Auto-fit grids narrow their first cards as later panels are added.
  // Recalculate plot sizes once every card has reached its final width.
  await Promise.all(plots.map(chart=>window.Plotly.Plots.resize(chart)));
  if(version===renderVersion)$('plots').setAttribute('aria-busy','false');
}

function showError(error){message(error.message || String(error),true);$('plots').setAttribute('aria-busy','false');}
async function displayDataset(loaded,source) {
  dataset=loaded;entries=loaded.rows.map((row,i)=>({row,index:i+1}));zoom=null;
  const resets=new Map();entries.forEach(e=>{const r=e.row.beac_num_sc_resets;if(r!==null&&r!==undefined)resets.set(String(r),(resets.get(String(r))||0)+1);});
  $('reset-filter').replaceChildren(new Option('All resets','all'));
  [...resets].sort((a,b)=>Number(a[0])-Number(b[0])).forEach(([key,count])=>$('reset-filter').add(new Option(`Reset ${key} · ${number(count)} samples`,key)));
  $('reset-filter').disabled=!resets.size;$('axis').value='record';
  [...$('axis').options].forEach(o=>{o.disabled=o.value==='boot'?!loaded.columns.includes('beac_time_since_boot'):o.value==='header'?!loaded.columns.includes('ccsdsSecHeader2_sec_beacon')||!loaded.columns.includes('ccsdsSecHeader2_sub_beacon'):false;});
  $('filename').textContent=source.name;
  $('source-detail').textContent=mode==='local'?`${(source.size/1024/1024).toFixed(1)} MB · ${loaded.tableName} · ${loaded.catalog.length} packet types · DuckDB ${loaded.dbVersion}`:`${(source.size/1024/1024).toFixed(1)} MB download · curated beacon snapshot · ${number(loaded.rows.length)} samples`;
  document.querySelector('.source-strip .pill').textContent=mode==='local'?'READ ONLY':'PUBLIC SNAPSHOT';
  $('welcome').hidden=true;$('workspace').hidden=false;$('export').disabled=false;
  $('range-note').textContent='Drag a plot to zoom every panel.';
  navigation();await render();
  const failed=loaded.rows.filter(r=>r.decode_status&&r.decode_status!=='decoded').length;
  message(failed?`${number(failed)} rows have a decode status other than “decoded”. Their stored values remain visible; missing values are gaps.`:'');
}
async function openFile(file) {
  if(!file || loading || mode!=='local') return;
  loading=true;$('open-file').disabled=true;$('welcome-open').disabled=true;
  message(`Opening ${file.name} locally…`);
  try{
    await getPlotly();
    const loaded=await loadDatabase(file,status=>message(status));
    // Commit a new dataset only after validation succeeds; keep the prior view on errors.
    await displayDataset(loaded,{name:file.name,size:file.size});
  }catch(error){showError(error);}
  finally{loading=false;$('open-file').disabled=false;$('welcome-open').disabled=false;$('file-input').value='';}
}
function exportCSV() {
  if(!dataset)return;
  const visiblePairs=POWER_PAIRS.filter(p=>(dataset.columns.includes(p.voltage)||dataset.columns.includes(p.current))&&pairMatches(p));
  const pairs=activeGroup==='electrical'||activeGroup==='overview'&&!$('search').value.trim()?visiblePairs.filter(p=>dataset.columns.includes(p.voltage)&&dataset.columns.includes(p.current)):[];
  const visiblePairFields=new Set(visiblePairs.flatMap(p=>[p.voltage,p.current]));
  const allPairFields=new Set(POWER_PAIRS.flatMap(p=>[p.voltage,p.current]));
  const signals=activeGroup==='electrical'?SIGNALS.filter(s=>available(s)&&s.group==='electrical'&&(visiblePairFields.has(s.field)||!allPairFields.has(s.field)&&matches(s))):selectedSignals(activeGroup==='overview'?null:activeGroup);
  const view=getEntries().filter(e=>!zoom || e.x>=zoom[0]&&e.x<=zoom[1]);
  // Spreadsheet-safe strings. Numeric negatives remain numbers, not formulas.
  const csvCell=v=>{if(v===null||v===undefined)return '';let s=String(v);if(typeof v==='string'&&/^[=+\-@\t\r]/.test(s))s=`'${s}`;return /[",\r\n]/.test(s)?`"${s.replaceAll('"','""')}"`:s;};
  const lines=[['beacon_sample',axisTitle(),...signals.map(s=>`${s.field}${s.unit?` [${s.unit}]`:''}`),...pairs.map(p=>`${p.id}_power [W]`)].map(csvCell).join(',')];
  for(const entry of view){const derived=pairs.map(p=>powerValue(entry.row,p));lines.push([entry.index,entry.x,...signals.map(s=>value(entry.row,s)),...derived].map(csvCell).join(','));}
  const url=URL.createObjectURL(new Blob([lines.join('\r\n')],{type:'text/csv;charset=utf-8'}));const link=element('a');link.href=url;link.download=`${dataset.filename.replace(/\.[^.]+$/,'')}-${activeGroup}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

$('open-file').onclick=$('welcome-open').onclick=()=>$('file-input').click();
$('file-input').onchange=()=>openFile($('file-input').files[0]);
$('reset-zoom').onclick=()=>setZoom(null);
$('export').onclick=exportCSV;
for(const id of ['axis','reset-filter','duplicates'])$(id).onchange=()=>{zoom=null;$('range-note').textContent='Drag a plot to zoom every panel.';render().catch(showError);};
$('y-scale').onchange=()=>render().catch(showError);
let searchTimer;$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>render().catch(showError),250);};
let dragDepth=0;
window.addEventListener('dragenter',e=>{if(mode==='local'&&e.dataTransfer.types.includes('Files')){e.preventDefault();dragDepth++;$('drop-overlay').hidden=false;}});
window.addEventListener('dragover',e=>{if(e.dataTransfer.types.includes('Files'))e.preventDefault();});
window.addEventListener('dragleave',()=>{dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)$('drop-overlay').hidden=true;});
window.addEventListener('drop',e=>{e.preventDefault();dragDepth=0;$('drop-overlay').hidden=true;openFile(e.dataTransfer.files[0]);});
navigation();

function setMode(nextMode) {
  mode=nextMode;document.body.dataset.mode=mode;
  if(mode==='public'){
    document.querySelector('.sidebar-bottom').replaceChildren(element('span','local-dot'),document.createTextNode('Public telemetry'),element('small','','Browse published captures from the SunCET telemetry archive.'));
    document.querySelector('#welcome .eyebrow').textContent='SUNCET TELEMETRY ARCHIVE';
    document.querySelector('#welcome h2').textContent='A closer look at every orbit.';
    document.querySelector('#welcome > p').textContent='Explore published spacecraft captures, from thermal behavior and electrical power to operating modes.';
    document.querySelector('.welcome-detail').textContent='Choose a published capture above to get started.';
    $('page-description').textContent='Explore the published spacecraft telemetry archive.';
  }
}
async function openCapture(item) {
  if(loading)return;
  loading=true;$('capture').disabled=true;$('export').disabled=true;
  try{
    message(`Loading ${item.title}…`);await getPlotly();
    const loaded=await loadSnapshot(item.url,status=>message(status));
    await displayDataset(loaded,{name:item.title,size:loaded.byteSize || item.bytes || 0});
    loadedCaptureId=item.id;
    const url=new URL(location.href);url.searchParams.set('capture',item.id);history.replaceState(null,'',url);
  }catch(error){if(loadedCaptureId)$('capture').value=loadedCaptureId;showError(error);}
  finally{loading=false;$('capture').disabled=false;$('export').disabled=!dataset;}
}
async function loadCatalog(catalogURL) {
  const url=new URL(catalogURL,location.href);
  if(!['http:','https:'].includes(url.protocol))throw new Error('The catalog must use HTTP or HTTPS.');
  message('Loading the telemetry archive…');
  let response;
  try{response=await fetch(url,{credentials:'omit'});}
  catch{throw new Error('Could not download the capture catalog. Check its public URL and allow this dashboard origin in the data host’s CORS settings.');}
  if(!response.ok)throw new Error(`Could not load the telemetry catalog (${response.status}).`);
  const catalog=await response.json();
  if(catalog.format!=='suncet-catalog-v1'||!Array.isArray(catalog.datasets)||catalog.datasets.length>1000)throw new Error('This URL does not contain a supported SunCET capture catalog.');
  archive=catalog.datasets.map(item=>{
    if(!item || typeof item.id!=='string'||typeof item.title!=='string'||typeof item.url!=='string')throw new Error('The capture catalog contains an invalid entry.');
    const snapshotURL=new URL(item.url,response.url);
    if(!['http:','https:'].includes(snapshotURL.protocol))throw new Error('Capture URLs must use HTTP or HTTPS.');
    const date=(item.id.match(/\d{4}-\d{2}-\d{2}/)||snapshotURL.pathname.match(/\d{4}-\d{2}-\d{2}/))?.[0];
    const title=date?item.title.replace(/^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?(?=\s*[·—–|:]|$)/i,date):item.title;
    return {...item,title,url:snapshotURL.href};
  });
  if(!archive.length){message('No captures have been published yet.');return;}
  $('archive-picker').hidden=false;$('capture').replaceChildren(...archive.map(item=>new Option(item.title,item.id)));
  const id=new URLSearchParams(location.search).get('capture');
  const chosen=archive.find(item=>item.id===id)||archive[0];$('capture').value=chosen.id;
  $('capture').onchange=()=>openCapture(archive.find(item=>item.id===$('capture').value));
  await openCapture(chosen);
}
async function startup() {
  const response=await fetch('./site-config.json');
  if(!response.ok)throw new Error('The dashboard configuration could not be loaded.');
  const config=await response.json();
  const previewCatalog=new URLSearchParams(location.search).get('catalog');
  setMode(previewCatalog?'public':config.mode==='local'?'local':'public');
  if(mode==='public'){
    if(previewCatalog||config.catalogUrl)await loadCatalog(previewCatalog||config.catalogUrl);
    else message('The public telemetry archive has not been connected yet.');
  }else{
    const response=await fetch('./local-config.json');const local=response.ok?await response.json():null;
    if(local?.url){
      message(`Loading ${local.filename}…`);const response=await fetch(local.url);
      if(!response.ok)throw new Error(`Could not load dataset (${response.status}).`);
      const blob=await response.blob();await openFile(new File([blob],local.filename,{type:'application/octet-stream'}));
    }
  }
}
startup().catch(showError);
