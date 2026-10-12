"""Generate a portable, dependency-free, interactive comparison report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>musaProbe 精度分析报告</title>
<style>
:root{color-scheme:light;--ink:#142536;--muted:#516478;--line:#d9e2ec;--paper:#fff;--blue:#1559a6;--bg:#f2f6fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,sans-serif}
header{background:#102d49;color:#fff;padding:24px 32px}header h1{font-size:27px;margin:0}header p{margin:5px 0 0;color:#cfdeec}
main{padding:22px 28px;max-width:1800px;margin:auto}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{background:var(--paper);border:1px solid var(--line);border-radius:9px;padding:14px 20px;min-width:130px}.card strong{display:block;font-size:23px}.card span{color:var(--muted)}
.notice{padding:14px 18px;border-left:4px solid #d29329;background:#fff8e8;margin:18px 0}.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:18px 0}input,select,button{font:inherit;border:1px solid #b6c4d3;border-radius:5px;background:#fff;color:var(--ink);padding:7px 9px}input[type=search]{min-width:250px;flex:1}button{cursor:pointer}button:hover{background:#e9f1fb}label{white-space:nowrap}.layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:16px}.panel{background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden}.panel h2{font-size:15px;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}
#tree{padding:12px;max-height:650px;overflow:auto}#tree details{padding-left:10px}#tree summary{cursor:pointer;overflow-wrap:anywhere}#tree button{padding:1px 5px;border:0;text-align:left;font-size:12px;background:transparent}.table-wrap{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{position:sticky;top:0;background:#eaf0f7;white-space:nowrap;z-index:1}td.path{min-width:180px;max-width:420px;overflow-wrap:anywhere}tr.data-row{cursor:pointer}tr.data-row:hover{background:#edf5ff}tr.selected{background:#e3efff}.pill{font-size:11px;border-radius:12px;padding:3px 8px;display:inline-block;white-space:nowrap;background:#e9edf2}.pass{background:#d9f2e3;color:#146d3b}.mismatch,.nonfinite,.error,.shape_mismatch,.dtype_mismatch,.unaligned{background:#ffe1dc;color:#9f281d}.statistics_only,.truncated,.unsupported,.missing_reference,.missing_candidate{background:#fff0c8;color:#8a5900}
#details{margin-top:16px;padding:18px}#details h2{font-size:18px;margin:0 0 12px}#details pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,monospace;background:#f3f6fa;padding:12px;border-radius:6px}#details .detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}#pager{display:flex;align-items:center;gap:10px;padding:10px 14px}#coverage{margin-top:16px;padding:16px}#coverage pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.empty{padding:22px;color:var(--muted)}.subtle{color:var(--muted)}.legend{font-size:12px;color:var(--muted);margin:10px 0}.mono{font-family:ui-monospace,monospace}footer{padding:20px 0;color:var(--muted);font-size:12px}@media(max-width:800px){main{padding:12px}.layout{grid-template-columns:1fr}#tree{max-height:200px}#details .detail-grid{grid-template-columns:1fr}header{padding:20px}.cards .card{flex:1}}
</style></head><body>
<header><h1>musaProbe</h1><p>精度分析报告 · 离线浏览 · 无网络依赖</p></header>
<main><div class="cards" id="cards"></div>
<div class="notice" id="notice"></div>
<div class="filters"><input type="search" id="search" aria-label="搜索模块算子或张量" placeholder="搜索模块、算子、张量路径、原因…"><select id="status" aria-label="状态筛选"><option value="">全部状态</option></select><select id="rank" aria-label="rank 筛选"><option value="">全部 rank</option></select><label><input id="problems" type="checkbox"> 仅异常和未完成</label><button id="reset">重置筛选</button></div>
<div class="legend">pass：真实张量通过；statistics_only：仅有统计量；incomplete：覆盖不足。点击记录查看指标和调用栈。</div>
<div class="layout"><aside class="panel"><h2>模型层级 <button id="allmodules">全部模块</button></h2><div id="tree"></div></aside>
<section class="panel"><h2 id="results-title">记录</h2><div class="table-wrap"><table><thead><tr><th>状态</th><th>Rank / Step / μbatch</th><th>模块 / 张量</th><th>阶段 / 类型</th><th>最大绝对误差</th><th>超阈值比例</th></tr></thead><tbody id="rows"></tbody></table><div id="empty" class="empty" hidden>没有符合筛选条件的记录。</div></div><div id="pager"><button id="prev">上一页</button><span id="pageinfo"></span><button id="next">下一页</button></div></section></div>
<section class="panel" id="details"><h2>记录详情</h2><p class="subtle">选择一条记录查看对齐身份、指标、采集状态和调用栈。</p></section>
<details class="panel" id="coverage"><summary>采集覆盖与运行信息</summary><pre id="coverage-content"></pre></details>
<noscript><p>交互筛选需要浏览器启用 JavaScript。完整机器可读结果在同目录 comparison.json / comparison.csv。</p></noscript>
<footer>首处观测差异是排查入口，不能据此认定根因。报告只反映已采集的范围，不代表模型全部计算均已验证。</footer></main>
<script id="comparison-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('comparison-data').textContent);
const all=data.records||[], $=id=>document.getElementById(id);
let filtered=[], page=0, selectedModule=null, selected=null;
const pageSize=100;
function element(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);if(cls)e.className=cls;return e;}
function val(v){if(v===null||v===undefined)return '—';if(typeof v==='number'&&!Number.isInteger(v))return v.toPrecision(5);return String(v);}
function pretty(v){return JSON.stringify(v,null,2);}
function gradientLayout(r){if(r.gradient_layout)return r.gradient_layout;const fields=['gradient_scope','original_parameter_shape','parameter_flat_range','gradient_stage','gradient_parallel_coordinates'];const layout={};for(const side of ['reference','candidate']){const event=r[side]||{};layout[side]=Object.fromEntries(fields.filter(key=>Object.prototype.hasOwnProperty.call(event,key)).map(key=>[key,event[key]]));}return layout;}
function badge(status){return element('span',status,'pill '+status.replace(/[^a-z_]/g,''));}
const counts=(data.summary||{}).statuses||{};
for(const [label,value] of [['总体状态',data.overall],['比较记录',all.length],['真实张量通过',counts.pass||0],['含NaN/Inf记录',counts.nonfinite||0]]){const c=element('div',undefined,'card');c.append(element('strong',value),element('span',label));$('cards').append(c);}
const first=data.first_difference;
if(first){const id=first.identity||{};$('notice').textContent='首处观测差异：rank '+val(first.rank)+' / step '+val(id.step)+' / '+(id.module||'<root>')+' / '+(id.tensor_path||'')+' ['+first.status+']。这不是根因判定。';}
else if(data.overall==='fail'){$('notice').textContent='采集或对齐存在错误，尚未形成有效的张量差异定位。请查看 error / unaligned 记录；不能据此认定模型数值异常。';}
else{$('notice').textContent=data.overall==='pass'?'当前采集范围内的真实张量均满足比较规则。':'未发现已确认的张量差异，但采集或对齐覆盖不完整；不能据此宣称精度通过。';}
for(const status of Object.keys(counts).sort()){const o=element('option',status+' ('+counts[status]+')');o.value=status;$('status').append(o);}
for(const rank of [...new Set(all.map(r=>r.rank))].sort((a,b)=>(a??-1)-(b??-1))){const o=element('option','rank '+val(rank));o.value=String(rank);$('rank').append(o);}
const tree={name:'模型',path:null,children:new Map(),count:0,issues:0};
for(const r of all){const module=(r.identity||{}).module||'';const names=module?module.split('.'):['<root>'];let node=tree;node.count++;if(r.status!=='pass')node.issues++;let path=[];for(const name of names){path.push(name);if(!node.children.has(name))node.children.set(name,{name,path:module?path.join('.'):'' ,children:new Map(),count:0,issues:0});node=node.children.get(name);node.count++;if(r.status!=='pass')node.issues++;}}
function renderNode(node,parent){for(const child of [...node.children.values()].sort((a,b)=>a.name.localeCompare(b.name))){const details=element('details');details.open=node===tree;const summary=element('summary');summary.append(element('span',child.name+' ('+child.count+', 待查 '+child.issues+') '));const choose=element('button','筛选');choose.addEventListener('click',e=>{e.preventDefault();e.stopPropagation();selectedModule=child.path;refresh();});summary.append(choose);details.append(summary);renderNode(child,details);parent.append(details);}}
renderNode(tree,$('tree'));
function refresh(){const query=$('search').value.trim().toLowerCase(), status=$('status').value, rank=$('rank').value, problems=$('problems').checked;filtered=all.filter(r=>{const id=r.identity||{},module=id.module||'';return(!status||r.status===status)&&(!rank||String(r.rank)===rank)&&(!problems||r.status!=='pass')&&(selectedModule===null||module===selectedModule||module.startsWith(selectedModule+'.'))&&(!query||JSON.stringify([id,r.status,r.reason]).toLowerCase().includes(query));});page=0;renderRows();}
function renderRows(){const body=$('rows');body.replaceChildren();const slice=filtered.slice(page*pageSize,(page+1)*pageSize);for(const r of slice){const id=r.identity||{},m=r.metrics||{},tr=element('tr',undefined,'data-row'+(r===selected?' selected':''));const state=element('td');state.append(badge(r.status));const path=element('td',undefined,'path');path.append(element('div',id.module||'<root>','mono'),element('div',id.tensor_path||'', 'subtle'));if(id.op)path.append(element('div',id.op,'subtle'));tr.append(state,element('td',val(r.rank)+' / '+val(id.step)+' / '+val(id.microbatch)),path,element('td',(id.phase||'—')+' / '+(id.kind||'—')),element('td',val(m.max_abs_error)),element('td',m.mismatch_ratio==null?'—':(m.mismatch_ratio*100).toPrecision(4)+'%'));tr.addEventListener('click',()=>{selected=r;showDetails(r);renderRows();});body.append(tr);}$('empty').hidden=filtered.length!==0;$('results-title').textContent='记录 '+filtered.length+' / '+all.length+(selectedModule===null?'':' · '+(selectedModule||'<root>'));$('pageinfo').textContent=(filtered.length?page+1:0)+' / '+Math.ceil(filtered.length/pageSize)+' 页';$('prev').disabled=page===0;$('next').disabled=(page+1)*pageSize>=filtered.length;}
function showDetails(r){const p=$('details');p.replaceChildren();const h=element('h2','记录详情 ');h.append(badge(r.status));p.append(h,element('p',r.reason||''));p.append(element('h3','语义身份'),element('pre',pretty({rank:r.rank,...r.identity})),element('h3','误差指标'),element('pre',pretty(r.metrics||{})));if(r.threshold_policy||r.mapping||r.fp8_quantization){p.append(element('h3','阈值与显式映射证据'),element('pre',pretty({threshold_policy:r.threshold_policy,mapping:r.mapping,fp8_quantization:r.fp8_quantization}))); }const layout=gradientLayout(r);if(Object.values(layout).some(value=>Object.keys(value).length)){p.append(element('h3','梯度分片与采集阶段'),element('pre',pretty(layout)));}const grid=element('div',undefined,'detail-grid');for(const [key,label] of [['reference','参考运行'],['candidate','候选运行']]){const box=element('div');box.append(element('h3',label));const event=r[key];if(event){const meta={...event};delete meta.stack;box.append(element('pre',pretty(meta)),element('h4','调用栈'));const stack=Array.isArray(event.stack)?event.stack:[];box.append(element('pre',stack.length?stack.map(frame=>typeof frame==='string'?frame:val(frame.file)+':'+val(frame.line)+'  '+val(frame.function)).join('\n'):'未采集调用栈'));}else box.append(element('p','该运行没有对应记录。'));grid.append(box);}p.append(grid);}
$('coverage-content').textContent=pretty({reference:data.reference,candidate:data.candidate,summary:data.summary,coverage:data.coverage,notes:data.notes});
for(const id of ['search','status','rank','problems'])$(id).addEventListener(id==='search'?'input':'change',refresh);
$('reset').addEventListener('click',()=>{$('search').value='';$('status').value='';$('rank').value='';$('problems').checked=false;selectedModule=null;refresh();});
$('allmodules').addEventListener('click',()=>{selectedModule=null;refresh();});$('prev').addEventListener('click',()=>{if(page>0)page--;renderRows();});$('next').addEventListener('click',()=>{if((page+1)*pageSize<filtered.length)page++;renderRows();});
refresh();if(first){selected=first;showDetails(first);}
</script></body></html>'''


def _browser_safe_integers(value: Any) -> Any:
    """Keep int64 metadata exact in browsers, whose JSON numbers are doubles."""
    if type(value) is int and abs(value) > (2 ** 53 - 1):
        return str(value)
    if isinstance(value, dict):
        return {key: _browser_safe_integers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_browser_safe_integers(item) for item in value]
    return value


def render_report(comparison: dict[str, Any], output_file: str | Path) -> str:
    """Write a self-contained HTML report, safely embedding untrusted labels.

    Integers beyond JavaScript's safe range are displayed as decimal strings;
    the source comparison and its JSON/CSV outputs are not modified.
    JSON characters capable of closing a script element are encoded. The UI
    creates DOM nodes with textContent and never interprets capture strings as
    HTML, selectors, URLs, or executable script.
    """
    payload = json.dumps(_browser_safe_integers(comparison), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    output_path = Path(output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    return str(output_path.resolve())


_ANALYSIS_STYLE = '''body{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#173149;background:#f2f6fa}h1{font-size:26px}select,input,button{font:inherit;padding:7px;max-width:95%;margin:5px}svg{background:white;border:1px solid #c9d5df;max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}table{border-collapse:collapse;background:white;width:100%}td,th{text-align:left;border-bottom:1px solid #d9e2ec;padding:8px}#viewport{overflow:auto;max-height:650px}.notice{border-left:4px solid #c28b29;padding:12px;background:#fff4db}circle,rect{cursor:pointer}'''


def _render_analysis(data, output_file, title, body, script):
    payload = json.dumps(_browser_safe_integers(data), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for char, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(char, escaped)
    content = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               f'<title>musaProbe {title}</title><style>{_ANALYSIS_STYLE}</style><h1>musaProbe {title}</h1>'
               '<p class="notice" id="scope"></p>' + body + '<h2>详情</h2><pre id="detail"></pre>'
               '<details><summary>完整证据</summary><pre id="evidence"></pre></details>'
               '<script id="analysis-data" type="application/json">' + payload + '</script><script>\n'
               "'use strict';const data=JSON.parse(document.getElementById('analysis-data').textContent),$=id=>document.getElementById(id);"
               "$('scope').textContent=data.scope;$('evidence').textContent=JSON.stringify(data,null,2);"
               "function el(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);return e;}"
               "function svg(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs||{}))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=String(text);return e;}"
               + script + '\n</script></html>')
    target = Path(output_file).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target.resolve())


def render_trends(data, output_file):
    """Interactive step chart; unavailable samples explicitly break the line."""
    body = '<label>序列 <select id="series"></select></label><label>指标 <select id="metric"></select></label><p id="window"></p><div id="viewport"><svg id="chart" viewBox="0 0 1100 400" width="1100" height="400" role="img" aria-label="统计趋势与缺口"></svg></div><table><thead><tr><th>Step</th><th>状态</th><th>数值</th></tr></thead><tbody id="points"></tbody></table>'
    script = r'''
for(const [i,s]of(data.series||[]).entries()){const o=el('option','rank '+s.rank+' / '+s.identity.module+' / '+s.identity.kind+' / '+s.identity.tensor_path);o.value=i;$('series').append(o);}
for(const m of ['mean','min','max','l2','nan_count','posinf_count','neginf_count']){const o=el('option',m);o.value=m;$('metric').append(o);}
function draw(){const s=(data.series||[])[$('series').value],chart=$('chart');chart.replaceChildren();$('points').replaceChildren();if(!s){$('detail').textContent='没有可绘制序列。';return;}$('detail').textContent=JSON.stringify(s.identity,null,2);const metric=$('metric').value,points=(s.points||[]).slice(0,2000);$('window').textContent='显示前 '+points.length+' / '+(s.points||[]).length+' 个点；完整数据保留在 JSON。超出浏览器安全整数范围的数值仅以文本显示。';let finite=points.filter(p=>['available','nonfinite'].includes(p.state)&&typeof p.metrics[metric]==='number'&&Number.isFinite(p.metrics[metric]));const values=finite.map(p=>p.metrics[metric]);let lo=values.reduce((a,b)=>Math.min(a,b),Infinity),hi=values.reduce((a,b)=>Math.max(a,b),-Infinity);if(!values.length){lo=0;hi=1;}if(lo===hi){lo-=1;hi+=1;}const minStep=points.length?points[0].step:0,maxStep=points.length?points[points.length-1].step:1;const x=p=>70+990*(p.step-minStep)/(maxStep-minStep||1),y=p=>350-300*(p.metrics[metric]-lo)/(hi-lo);chart.append(svg('line',{x1:70,y1:350,x2:1060,y2:350,stroke:'#71859b'}),svg('text',{x:5,y:45},hi.toPrecision(5)),svg('text',{x:5,y:350},lo.toPrecision(5)));let previous=null;for(const p of points){const tr=el('tr');tr.append(el('td',p.step),el('td',p.state),el('td',p.metrics[metric]??'—'));$('points').append(tr);const valid=['available','nonfinite'].includes(p.state)&&typeof p.metrics[metric]==='number'&&Number.isFinite(p.metrics[metric]);if(!valid){chart.append(svg('line',{x1:x(p),y1:365,x2:x(p),y2:380,stroke:'#c25525','stroke-width':3}));previous=null;continue;}if(previous)chart.append(svg('line',{x1:x(previous),y1:y(previous),x2:x(p),y2:y(p),stroke:'#1468ab','stroke-width':2}));const dot=svg('circle',{cx:x(p),cy:y(p),r:4,fill:p.state==='nonfinite'?'#bd3523':'#1468ab'});dot.append(svg('title',{},'step '+p.step+' / '+p.metrics[metric]));dot.addEventListener('click',()=>{$('detail').textContent=JSON.stringify(p,null,2);});chart.append(dot);previous=p;}chart.append(svg('text',{x:75,y:395},'缺口用底部短线表示；不插值，不补零。'));}
$('series').addEventListener('change',draw);$('metric').addEventListener('change',draw);draw();
'''
    return _render_analysis(data, output_file, "统计趋势", body, script)


def render_graph(data, output_file):
    """Display only graph edges present in evidence, with rank/search controls."""
    body = '<label>Rank <select id="rank"></select></label><input id="search" type="search" placeholder="搜索模块、算子或 Tensor ID"><p id="count"></p><div id="viewport"><svg id="chart" width="1200" height="500" role="img" aria-label="已观测数据依赖图"></svg></div>'
    script = r'''
const anyRank=el('option','全部 rank');anyRank.value='';$('rank').append(anyRank);const ranks=[...new Set((data.nodes||[]).map(n=>n.rank))].sort((a,b)=>a-b);for(const r of ranks){const o=el('option',r);o.value=r;$('rank').append(o);}function draw(){const query=$('search').value.toLowerCase(),rank=Number($('rank').value),all=(data.nodes||[]).filter(n=>$('rank').value===''||n.rank===rank),matches=all.filter(n=>!query||JSON.stringify([n.tensor_id,n.event.module,n.event.op]).toLowerCase().includes(query));const selected=matches.slice(0,300),ids=new Set(selected.map(n=>n.id)),edges=(data.edges||[]).filter(e=>ids.has(e.source)&&ids.has(e.target));$('count').textContent='当前显示 '+selected.length+' / '+matches.length+' 个匹配节点，'+edges.length+' 条已观察边；最多显示 300 节点。未知输入 '+(data.unknown_inputs||[]).filter(x=>$('rank').value===''||x.rank===rank).length+'。完整数据保留在 JSON。';const chart=$('chart');chart.replaceChildren();const defs=svg('defs',{}),marker=svg('marker',{id:'edge-arrow',markerWidth:7,markerHeight:7,refX:6,refY:3,orient:'auto'});marker.append(svg('path',{d:'M0,0 L0,6 L7,3 z',fill:'#91a9bf'}));defs.append(marker);chart.append(defs);chart.setAttribute('height',Math.max(300,selected.length*62+20));const positions=new Map();for(const[i,n]of selected.entries())positions.set(n.id,{x:180+(i%3)*300,y:30+i*62});for(const edge of edges){const a=positions.get(edge.source),b=positions.get(edge.target);chart.append(svg('path',{d:'M '+(a.x+105)+' '+(a.y+20)+' C 1120 '+a.y+',1120 '+b.y+','+(b.x+105)+' '+b.y,fill:'none',stroke:'#91a9bf','stroke-width':1,'marker-end':'url(#edge-arrow)'}));}for(const n of selected){const p=positions.get(n.id),g=svg('g',{}),bad=Object.values(n.nonfinite||{}).some(v=>v>0),r=svg('rect',{x:p.x,y:p.y,width:270,height:36,rx:5,fill:bad?'#ffe1dc':'#e4f0fa',stroke:bad?'#ba3f29':'#456c93'});g.append(r,svg('text',{x:p.x+8,y:p.y+23,'font-size':12},n.event.op||n.tensor_id));g.append(svg('title',{},n.event.module+' / '+n.tensor_id));g.addEventListener('click',()=>{$('detail').textContent=JSON.stringify({node:n,incoming:edges.filter(e=>e.target===n.id),outgoing:edges.filter(e=>e.source===n.id),unknown_inputs:(data.unknown_inputs||[]).filter(e=>e.target===n.id)},null,2);});chart.append(g);}}
$('rank').addEventListener('change',draw);$('search').addEventListener('input',draw);$('detail').textContent=JSON.stringify({summary:data.summary,status:data.status,observed_nonfinite_boundaries:data.observed_nonfinite_boundaries,errors:data.errors},null,2);draw();
'''
    return _render_analysis(data, output_file, "已观测数据依赖图", body, script)
