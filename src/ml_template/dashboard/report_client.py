"""Inline clientside capture, packaged as Python so no assets configuration is needed."""

CLIENTSIDE_CAPTURE = r"""
function(htmlClicks, zipClicks, tab, consent) {
    "use strict";
    const ids={assets:"assets-container",sweeps:"sweeps-container",experiments:"experiments-container",run:"single-run-container",compare:"compare-container",calibration:"calibration-container",evaluation:"evaluation-container"};
    const ctx=window.dash_clientside.callback_context || {};
    const trigger=String(ctx.triggered_id || (((ctx.triggered||[])[0]||{}).prop_id)||"").split(".")[0];
    if (!["report-export-html","report-export-zip"].includes(trigger)) return window.dash_clientside.no_update;
    const format=trigger==="report-export-zip"?"zip":"html";
    function visible(n) {
        if (!n || n.nodeType!==1) return true;
        const s=window.getComputedStyle(n);
        return s.display!=="none" && s.visibility!=="hidden" && s.visibility!=="collapse" && s.opacity!=="0" && (s.display==="contents" || n.getClientRects().length>0);
    }
    function text(n) {
        if(n.nodeType===3) return n.nodeValue||"";
        if(n.nodeType!==1 || !visible(n) || ["SCRIPT","STYLE","NOSCRIPT","BUTTON","INPUT","SELECT","OPTION"].includes(n.tagName)) return "";
        return Array.from(n.childNodes).map(text).join("");
    }
    function plain(v) {return JSON.parse(JSON.stringify(v,function(k,x){if(k.startsWith("_"))return undefined;if(ArrayBuffer.isView(x))return Array.from(x);if(typeof x==="bigint")throw new Error("BigInt chart data is unsupported");return x;}));}
    try {
        if(!Object.prototype.hasOwnProperty.call(ids,tab)) throw new Error("Unknown tab");
        const allowed=Array.isArray(consent)&&consent.includes("allow_final_evaluation");
        if(tab==="evaluation"&&!allowed) throw new Error("Explicitly allow exporting visible Final evaluation content first");
        const root=document.getElementById(ids[tab]);
        if(!root||!visible(root)) throw new Error("Selected view is not rendered; wait and retry");
        if(root.matches('[data-dash-is-loading="true"]')||root.querySelector('[data-dash-is-loading="true"]')) throw new Error("View is still loading; wait and retry");
        const blocks=[];let heading="",visited=0;
        function grid(n) {
            let captured=0;
            [["center",".ag-center-cols-container",".ag-header-viewport"],["pinned left",".ag-pinned-left-cols-container",".ag-pinned-left-header"],["pinned right",".ag-pinned-right-cols-container",".ag-pinned-right-header"]].forEach(function(p){
                const body=n.querySelector(p[1]),header=n.querySelector(p[2]);
                if(!body||!visible(body))return;
                const rr=Array.from(body.querySelectorAll('[role="row"]')).filter(visible);
                if(!rr.length)return;
                let cols=header?Array.from(header.querySelectorAll('[role="columnheader"]')).filter(visible):[];
                if(!cols.length)cols=Array.from(rr[0].querySelectorAll('[role="gridcell"]')).filter(visible);
                cols.sort((a,b)=>a.getBoundingClientRect().left-b.getBoundingClientRect().left);
                const keys=cols.map(c=>c.getAttribute("col-id"));
                if(!keys.length||keys.some(k=>k===null)||new Set(keys).size!==keys.length)throw new Error("Grid columns are not ready");
                const rows=[["AG row index"].concat(cols.map(c=>{const label=c.querySelector(".ag-header-cell-text");return label?text(label).trim():c.getAttribute("col-id");}))];
                rr.sort((a,b)=>a.getBoundingClientRect().top-b.getBoundingClientRect().top).forEach(function(row){
                    const cells=new Map(Array.from(row.querySelectorAll('[role="gridcell"]')).filter(visible).map(c=>[c.getAttribute("col-id"),text(c).trim()]));
                    if(!keys.every(k=>cells.has(k)))throw new Error("Grid changed during capture; wait and retry");
                    rows.push([row.getAttribute("row-index")||row.getAttribute("aria-rowindex")||"not recorded"].concat(keys.map(k=>cells.get(k))));
                });
                blocks.push({kind:"table",caption:(heading||"AG Grid")+" — "+p[0]+" pane, rendered rows/columns only",rows:rows});captured++;
            });
            if(!captured) {
                if(n.querySelectorAll('[role="gridcell"]').length)throw new Error("Unsupported populated grid pane; no partial export made");
                blocks.push({kind:"paragraph",text:"AG Grid has no rendered data rows; no underlying rows exported."});
            }
            blocks.push({kind:"paragraph",text:"AG Grid captures rendered rows/columns only. Pinned column panes are separate; unrendered and pinned-top/bottom rows are excluded."});
        }
        function walk(n) {
            if(++visited>200000)throw new Error("View too large; reduce the selected content");
            if(n.nodeType===3){const value=(n.nodeValue||"").trim();if(value)blocks.push({kind:"paragraph",text:value});return;}
            if(n.nodeType!==1||!visible(n))return;
            const tag=n.tagName;
            if(["SCRIPT","STYLE","NOSCRIPT","SVG","CANVAS","BUTTON","INPUT","SELECT","OPTION","TEXTAREA"].includes(tag)||n.matches(".dash-dropdown, .Select-menu-outer"))return;
            if(n.classList.contains("ag-root")){grid(n);return;}
            if(n.classList.contains("js-plotly-plot")){
                if(!Array.isArray(n.data)||!n.layout)throw new Error("Chart is not ready");
                blocks.push({kind:"figure",figure:plain({data:n.data,layout:n.layout})});return;
            }
            if(tag==="TABLE"){
                const rows=Array.from(n.rows).filter(visible).map(r=>Array.from(r.cells).filter(visible).map(c=>text(c).trim())).filter(r=>r.length);
                if(rows.length)blocks.push({kind:"table",caption:heading,rows:rows});return;
            }
            if(/^H[1-6]$/.test(tag)){const value=text(n).trim();if(value){heading=value;blocks.push({kind:"heading",level:Number(tag.slice(1)),text:value});}return;}
            if(["P","PRE","LI","LABEL","SMALL","SUMMARY"].includes(tag)){const value=text(n).trim();if(value)blocks.push({kind:tag==="PRE"?"pre":"paragraph",text:value});return;}
            Array.from(n.childNodes).forEach(walk);
        }
        walk(root);
        if(!blocks.length)throw new Error("No rendered report content");
        const controls=[];
        root.querySelectorAll(".dash-dropdown").forEach(function(c){if(!visible(c))return;const selected=Array.from(c.querySelectorAll('.Select-value-label,[class*="-singleValue"],[class*="-multiValue__label"]')).filter(visible).map(x=>text(x).trim());controls.push({id:c.id||"dropdown",kind:"dropdown_display_labels",value:selected.join(" | ")||"No displayed selection"});});
        root.querySelectorAll('input[type="radio"]:checked,input[type="checkbox"]:checked').forEach(function(c){const label=c.closest("label")||c.parentElement;if(!label||!visible(label))return;const owner=c.closest("[id]");controls.push({id:owner?owner.id:"selection",kind:c.type,value:text(label).trim()||String(c.value)});});
        root.querySelectorAll('input[type="text"],input[type="number"],input[type="date"],textarea').forEach(function(c){if(visible(c)&&!c.closest(".dash-dropdown"))controls.push({id:c.id||"input",kind:c.type||"textarea",value:String(c.value)});});
        const result={schema_version:1,format:format,active_tab:tab,captured_at:new Date().toISOString(),include_final_evaluation:allowed,controls:controls,blocks:blocks,request_id:Date.now()+"-"+Math.random().toString(16).slice(2)};
        if(new TextEncoder().encode(JSON.stringify(result)).length>12*1024*1024)throw new Error("Snapshot exceeds 12 MiB; reduce view");
        return result;
    } catch(e) {return {error:String(e.message||e),active_tab:tab,format:format,request_id:Date.now()+"-"+Math.random()};}
}
"""
