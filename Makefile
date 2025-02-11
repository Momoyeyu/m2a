PY ?= .venv/bin/python
PY_ATST ?= .venv-atst/bin/python
SYS_PY ?= python3

MODEL ?= crnn
METHOD ?= m2a
ATTACK ?= mirage
MODE ?= single
# MODEL=crnn|atst_sed  METHOD=m2a|cw|faag|aro|custom
# ATTACK=mirage|mute   MODE=single|multi
CONFIG ?= configs/models/$(MODEL).yaml
SET ?=
JOBS ?= 1

SETS = --set model=$(MODEL) --set method=$(METHOD) --set attack_type=$(ATTACK) --set mode=$(MODE) $(foreach s,$(SET),--set $(s))
PY_FLAGS = --python crnn=$(abspath $(PY)) --python atst_sed=$(abspath $(PY_ATST))

.PHONY: sync sync-pip sync-atst env-conda attack attack-dry table1 table2 dry-table1 dry-table2 check clean

sync:                     ## 建环境（CRNN + 框架依赖；uv 优先，无 uv 退回 venv+pip）
	@if command -v uv >/dev/null 2>&1; then \
		uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python; \
	else \
		echo "未检测到 uv，改用 venv + pip"; \
		$(MAKE) --no-print-directory sync-pip; \
	fi

sync-pip:                 ## 没有 uv 时的兜底
	@command -v $(SYS_PY) >/dev/null 2>&1 || { echo "找不到 $(SYS_PY)"; exit 1; }
	$(SYS_PY) -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt

sync-atst:                ## 建 ATST-SED 环境（.venv-atst + ATST-SED/requirements.txt + pip -e）
	$(SYS_PY) -m venv .venv-atst
	$(PY_ATST) -m pip install -q --upgrade pip
	$(PY_ATST) -m pip install -q -r ATST-SED/requirements.txt
	$(PY_ATST) -m pip install -q -e ATST-SED

env-conda:                ## ATST-SED 上游 conda 脚本（有 conda 时用）
	cd ATST-SED && bash conda_create_environment.sh

attack:                   ## 跑单个 cell：MODEL x METHOD x ATTACK x MODE
	@command -v $(PY) >/dev/null 2>&1 || { echo "缺少 $(PY)：先 make sync"; exit 1; }
	$(PY) run_attack.py --config $(CONFIG) $(SETS)

attack-dry:               ## 只打印解析后的配置，不跑实验
	@command -v $(PY) >/dev/null 2>&1 || { echo "缺少 $(PY)：先 make sync"; exit 1; }
	$(PY) run_attack.py --config $(CONFIG) $(SETS) --dump

table1:                   ## 复现 Table 1（单目标），输出与表格同维度的 2-D 数组
	@command -v $(PY) >/dev/null 2>&1 || { echo "缺少 $(PY)：先 make sync"; exit 1; }
	$(PY) run_table.py --config configs/tables/table1_single.yaml --jobs $(JOBS) $(PY_FLAGS) $(foreach s,$(SET),--set $(s))

table2:                   ## 复现 Table 2（多目标），输出与表格同维度的 2-D 数组
	@command -v $(PY) >/dev/null 2>&1 || { echo "缺少 $(PY)：先 make sync"; exit 1; }
	$(PY) run_table.py --config configs/tables/table2_multi.yaml --jobs $(JOBS) $(PY_FLAGS) $(foreach s,$(SET),--set $(s))

dry-table1:               ## 只生成 table1 的 cell 命令，不执行
	$(PY) run_table.py --config configs/tables/table1_single.yaml --jobs $(JOBS) $(PY_FLAGS) $(foreach s,$(SET),--set $(s)) --dry-run

dry-table2:               ## 只生成 table2 的 cell 命令，不执行
	$(PY) run_table.py --config configs/tables/table2_multi.yaml --jobs $(JOBS) $(PY_FLAGS) $(foreach s,$(SET),--set $(s)) --dry-run

check:                    ## 字节码编译 m2a 包和两个入口脚本
	$(PY) -m py_compile m2a/*.py m2a/editors/*.py run_attack.py run_table.py && echo OK

clean:                    ## 删除生成的 results/ 和 logs/
	rm -rf results logs
