import json
import os
import sys
from pathlib import Path

os.environ['OPENVINO_TELEMETRY_DISABLED'] = '1'
sys.modules['openvino_telemetry'] = None
root = Path.cwd()
sys.path.insert(0, str(root / 'src'))
import openvino as ov
from hifimobinet import common97, n100_demo as demo

base = root / '.local/common97-integration-20261105'
old = root / 'results/edge-n100-openvino/2026-11-01/.local/originals/final/work-n100-clean-20261101T075458Z'
core = ov.Core()
catalog = demo.read(root / 'demo/n100_models.json')
rows = []
for mid in common97.MODEL_IDS:
    old_xml = old / mid / 'int8.xml'
    old_model = core.read_model(str(old_xml))
    old_fq = sum(n.get_type_name() == 'FakeQuantize' for n in old_model.get_ops())
    try:
        common97.audit_graph(old_model, mid)
    except ValueError as error:
        rejected = str(error)
    else:
        raise AssertionError('old graph passed new validation')
    new_xml = base / 'work' / mid / 'int8.xml'
    new_audit = common97.audit_graph(core.read_model(str(new_xml)), mid)
    prep = demo.read(new_xml.parent / 'preparation.json')
    item = next(m.copy() for m in catalog['models'] if m['id'] == mid)
    item['artifacts'] = {'int8': prep['artifacts']['int8']['files']}
    item['int8_recipe'] = common97.RECIPE
    demo.check_common97_preparation(prep, item)
    rows.append({'model': mid, 'old_graph_sha256': demo.sha(old_xml),
                 'old_fq_count': old_fq, 'old_rejected': rejected,
                 'new_graph_sha256': demo.sha(new_xml), 'new_fq_count': new_audit['fake_quantize_count'],
                 'new_audit_passed': new_audit['passed'], 'demo_preparation_gate_passed': True})
result = {'status': 'passed', 'utc': demo.utc(), 'inference_performed_by_this_audit': False,
          'policy': common97.identity(), 'models': rows}
(base / 'legacy-and-new-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
