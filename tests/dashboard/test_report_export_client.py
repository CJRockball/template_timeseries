"""Node DOM-fixture checks; these are not browser acceptance."""

import json
import shutil
import subprocess

import pytest

from ml_template.dashboard.report_client import CLIENTSIDE_CAPTURE

NODE_FIXTURE = r"""

const assert=require('assert');const capture=__CLIENT__;
const T=x=>({nodeType:3,nodeValue:x});
function E(tag,children=[],opts={}){return {nodeType:1,tagName:tag,childNodes:children,data:opts.data,layout:opts.layout,rows:opts.rows||[],cells:opts.cells||[],hidden:!!opts.hidden,classList:{contains:x=>(opts.classes||[]).includes(x)},getClientRects(){return this.hidden?[]:[{}];},matches(s){return s.includes('data-dash-is-loading')&&!!opts.loading;},querySelector(s){return s.includes('data-dash-is-loading')&&opts.loading?this:null;},querySelectorAll(){return [];}};}
let root;global.window={getComputedStyle:n=>({display:n.hidden?'none':'block',visibility:'visible',opacity:'1'}),dash_clientside:{callback_context:{triggered_id:'report-export-html'},no_update:{}}};global.document={getElementById:()=>root};
let n=0;function check(f){f();n++;}
check(()=>{root=E('DIV',[E('P',[T('Visible')]),E('DIV',[E('P',[T('SECRET')])],{hidden:true})]);const s=capture(1,0,'experiments',[]);assert(!s.error);assert(!JSON.stringify(s).includes('SECRET'));});
check(()=>{root=E('DIV',[E('DIV',[],{classes:['js-plotly-plot'],data:[{x:new Float64Array([1,2]),y:[3,4]}],layout:{}})]);const s=capture(1,0,'compare',[]);assert.deepEqual(s.blocks[0].figure.data[0].x,[1,2]);});
check(()=>{root=E('DIV',[E('P',[T('Frozen')])]);assert(capture(1,0,'evaluation',[]).error);assert(!capture(1,0,'evaluation',['allow_final_evaluation']).error);});
check(()=>{root=E('DIV',[],{loading:true});assert(capture(1,0,'run',[]).error.includes('loading'));});
check(()=>{const c=E('TD',[T('[64, 32]')]);const row=E('TR',[c],{cells:[c]});root=E('DIV',[E('TABLE',[row],{rows:[row]})]);assert.equal(capture(1,0,'experiments',[]).blocks[0].rows[0][0],'[64, 32]');});
check(()=>{root=E('DIV',[E('P',[T('Ready')])]);window.dash_clientside.callback_context.triggered_id='report-export-zip';assert.equal(capture(1,1,'run',[]).format,'zip');});
console.log(JSON.stringify({checks_passed:n,browser_acceptance:false}));

"""


def test_capture_on_dom_fixture(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node not installed; perform browser acceptance")
    path = tmp_path / "capture.js"
    path.write_text(NODE_FIXTURE.replace("__CLIENT__", CLIENTSIDE_CAPTURE), encoding="utf-8")
    result = subprocess.run(
        [node, str(path)], capture_output=True, text=True, timeout=20, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["checks_passed"] == 6
