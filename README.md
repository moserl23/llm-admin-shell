# AI-Agent vs. Human System Administrators in Web Application Troubleshooting

This repository contains the code and experimental data for a master's thesis in cyber security that studies whether system logs generated during troubleshooting can distinguish **AI agents** from **human system administrators**.

The thesis is built around two controlled web application environments:

- **Nextcloud**
- **WordPress**

In each environment, faults are intentionally introduced, for example:

- missing pages
- internal server errors (`5xx`)
- broken buttons
- broken upload functionality
- configuration mistakes

Human participants and AI agents then attempt to diagnose and fix these problems. During the troubleshooting process, system activity is logged and later analyzed with statistical methods and machine learning pipelines.

The core research question is:

> Are the resulting logs sufficiently different to distinguish AI-driven troubleshooting from human troubleshooting?

## Quick Start

If you mainly want to explore the existing dataset and run the analysis code, this is the shortest path:

1. Extract the dataset from `data.zip`.
2. Install Python 3.12.3 (exact version used in this project), create and activate a virtual environment, and install the project dependencies via `pip install -r requirements.txt`
3. Inspect the aggregated datasets in `data/Nextcloud/combine/ExperimentAggregated/` or `data/WordPress/combine/ExperimentAggregated/`.
4. Run one baseline ML experiment.
5. Run one statistical experiment.
6. Inspect the generated CSV outputs and plots.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Extract the dataset archive:

```bash
unzip data.zip
```

If `unzip` is not available on your system, you can also use Python:

```bash
python3 -m zipfile -e data.zip .
```

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Example ML run:

```bash
python -m src.runners.ml.tfidf_360_nested \
  --dataset Nextcloud \
  --model svm \
  --log_type audit \
  --n_jobs 4 \
  --out_csv results/tfidf_nextcloud_audit_svm.csv
```

Example statistical run:

```bash
python -m src.runners.stats.one_gram_runner \
  --mode single \
  --dataset Nextcloud \
  --assignment_mode true \
  --log_type audit \
  --ngram_mode char \
  --metric js
```

If you want to reproduce the original troubleshooting data collection phase, first review the VM setup and agent sections below, as this stage depends on the thesis lab infrastructure.

## Repository Overview

The project consists of two main parts:

1. `agent/`
   Contains the code for the troubleshooting agents and the fault scenario definitions used in the experiments.

2. `src/`
   Contains the downstream analysis code:
   - machine learning pipelines
   - nested cross-validation runners
   - statistical analysis tools
   - null-hypothesis evaluation utilities
   - result ranking and comparison scripts

The `data/` directory contains the generated log dataset used by the analysis code.

## Project Structure

```text
.
├── agent/
│   ├── prompts/                 # Prompt material and in-context examples
│   ├── runners/                 # LLM, browser, and file-editing agents
│   ├── scenarios/
│   │   ├── nextcloud/           # Nextcloud fault scenarios
│   │   └── wordpress/           # WordPress fault scenarios
│   └── utils.py                 # SSH / sudo / logging utilities
├── data/
│   ├── Nextcloud/
│   └── WordPress/
├── src/
│   ├── analysis/                # Result analysis, ranking, null-hypothesis plots
│   ├── core/                    # Shared data loading, split generation, helpers
│   ├── ml_pipelines/            # Model-specific training/evaluation logic
│   ├── runners/
│   │   ├── ml/                  # Experiment entry points for ML pipelines
│   │   └── stats/               # Entry points for statistical tools
│   └── stats_tools/             # Statistical distance / complexity implementations
├── requirements.txt
└── README.md
```

## Experimental Setup

The experiments are based on two independent troubleshooting environments.

## VM Setup

The troubleshooting environment is based on a server-side LAMP setup. The same general stack is used throughout the experiments.

### Server VM

- OS: Ubuntu 24.04.2 LTS (`noble`)
- CPU / RAM: 2 vCPUs, 8 GB RAM (`8192 MB`)
- Web server: Apache `2.4.58` (Ubuntu build)
- Database: MariaDB `10.11.13`
- PHP: PHP `8.3.6`
- Stack model: always a classic **LAMP** stack

### Application versions

- Nextcloud: `31.0.7`
- WordPress: `6.9`

### Scenario model

The experiments are carried out in two separate application settings:

- Nextcloud deployed on a dedicated VM
- WordPress deployed on a dedicated VM

For each environment, faults are intentionally introduced and then solved either by:

- human participants
- AI agents

During these troubleshooting sessions, logs are recorded for later behavioral analysis.

### Logging infrastructure

The experiments rely on three main logging sources:

- Nextcloud OCC / admin audit logging into `nextcloud.log`
- Linux audit logging into `audit.log`
- standard Ubuntu system logging into `syslog`

#### Nextcloud OCC / admin audit logging

The following commands were used to enable and configure the relevant Nextcloud logging behavior:

```bash
php occ config:app:set admin_audit logfile --value=/var/www/nextcloud/data/nextcloud.log
sudo -u www-data php occ app:enable admin_audit
```

In addition, `/var/www/nextcloud/config/config.php` was modified with settings of the following form:

```php
'loglevel' => 2,
'log_type_audit' => 'file',
'logfile_audit' => '/var/www/nextcloud/data/nextcloud.log',
"log.condition" => [
  "apps" => ["admin_audit"],
],
```

This setup routes the relevant Nextcloud admin-audit events into:

- `/var/www/nextcloud/data/nextcloud.log`

#### Audit logging

Audit rules were deployed using Ansible via an automated configuration role from the public AIT Cybersecurity repository:

- [AIT Cybersecurity Repository](https://github.com/orgs/ait-cs-IaaS)
- [ansible-auditd role](https://github.com/ait-cs-IaaS/ansible-auditd)

This repository README does not reproduce those audit rules exactly; it states that the Linux audit subsystem used in the experiments was set up based on that external configuration source.

#### Syslog

Standard Ubuntu system logging via `syslog` was used in addition to the application-specific and audit logs.

### 1. Nextcloud scenario

- a virtual machine configured as a Linux server hosting a Nextcloud instance.
- a client VM for interaction
- intentionally injected faults
- recorded logs including:
  - `audit.log`
  - `syslog.log`
  - `nextcloud.log`

### 2. WordPress scenario

- a virtual machine configured as a Linux server hosting a WordPress instance.
- a client VM for interaction
- intentionally injected faults
- recorded logs including:
  - `audit.log`
  - `syslog.log`

## Agent Component

The `agent/` directory contains the automation used to perform troubleshooting actions against the target machines.

### Agent types

- `agent/runners/LLM_Agent.py`  
  Terminal-based troubleshooting agent for a Linux-based server hosting a Nextcloud instance on a LAMP stack
- `agent/runners/LLM_Agent_WP.py`  
  Terminal-based troubleshooting agent for a Linux-based server hosting a WordPress instance on a LAMP stack
- `agent/runners/browser_agent.py`
  Browser automation agent for Nextcloud
- `agent/runners/browser_agent_WP.py`
  Browser automation agent for WordPress
- `agent/runners/vim_agent.py`
  Specialized sub-agent for editing files via Vim-like patch reasoning

### Scenario definitions

The `agent/scenarios/` folder contains the fault injections used in the experiments.

#### Nextcloud scenarios

- `Break1.py`: broken trusted domains, login not possible
- `Break2.py`: broken database credentials, internal server error
- `Break3.py`: broken permissions on data directory, file operations fail
- `Break4.py`: missing application file, Files app breaks after login
- `Break5.py`: restrictive PHP memory limit
- `Break6.py`: Redis-related breakage, internal server error
- `Break7.py`: broken Apache `DocumentRoot`, malformed page behavior

#### WordPress scenarios

- `wp_break1.py`: broken base URL / navigation links return `404`
- `wp_break2.py`: broken database credentials
- `wp_break3.py`: broken media upload permissions
- `wp_break4.py`: deleted core file, fatal error / `500`
- `wp_break5.py`: syntax error in `wp-config.php`, `500`

Each scenario file contains:

- `config(session)` to introduce the fault
- `fix(session)` to restore the system to a working state

## Important Operational Assumptions

This is the most important section if you want to run the agent code on your own infrastructure.

The connection and privilege assumptions are implemented in [`agent/utils.py`](agent/utils.py).

### SSH and privilege requirements

The agent expects:

- SSH access to the target server **without an interactive password prompt**
- `sudo -i` to work **without an interactive password prompt**
- a shell environment in which the remote host can be controlled interactively via `pexpect`

In practice, the terminal agent logic assumes:

- passwordless SSH
- passwordless privilege escalation
- an interactive shell prompt that can be synchronized with a custom sentinel

### Current repository-specific behavior

At present, parts of the implementation are tailored to the experimental setup used in this project:

- `connect_root_setSentinel()` currently connects via the SSH alias `ssh arena_wp`
- log extraction relies on fixed paths defined in [`agent/utils.py`]
- both the Nextcloud and WordPress helper scripts currently share these configuration assumptions

As a consequence:

- the SSH target alias will need to be adapted for different environments  
- the log destination paths may need to be parameterized, especially when separating Nextcloud and WordPress logging outputs  

### Remote log paths currently assumed by the code

- `/var/www/nextcloud/data/nextcloud.log`
- `/var/log/syslog`
- `/var/log/auth.log`
- `/var/log/audit/audit.log`

These paths are initialized in `init_env_and_log_offsets(session)` and read by `read_new_logs(session)`.

## Environment Setup

### Python

Use Python 3.12.3 (exact version used in this project).

Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Node.js

Node.js is required for the browser agent, as the browser runners invoke npx @playwright/mcp@latest.

### Environment variables

Create a `.env` file in the repository root if needed.

Typical variables used by this project:

```env
OPENAI_API_KEY=...
DATAANALYSIS_DATA_ROOT=/absolute/path/to/custom/data/root
```

Notes:

- `OPENAI_API_KEY` is required for the LLM-based runners and the LLM-backed analysis pipeline.
- `DATAANALYSIS_DATA_ROOT` is optional and overrides the default dataset root used by the loaders.

## Data Layout

The repository contains both raw experiment logs and aggregated views.

### Raw per-run logs

The original troubleshooting sessions are stored per actor and per experiment, for example:

- `data/Nextcloud/LOGS_Result_<Actor>/expX/`
- `data/WordPress/LOGS_Result_<Actor>/expX/`

These folders contain the logs collected for individual troubleshooting runs.

### Aggregated views

The `combine/` directories provide derived views over the raw runs:

- `ExperimentAggregated/`
  concatenated per actor across experiments
- `PersonAggregated/`
  aggregated by experiment index and coarse label (`AI` vs `Human`)
- `AllAggregated/`
  aggregated globally by coarse label (`AI` vs `Human`)

### Default analysis target

The main loaders and experiment runners expect aggregated data under:

- `data/Nextcloud/combine/ExperimentAggregated/`
- `data/WordPress/combine/ExperimentAggregated/`

Each actor directory contains logs such as:

- `audit.log`
- `syslog.log`
- `nextcloud.log`

Notes:

- WordPress directories still keep a `nextcloud.log` slot for structural consistency, but the main WordPress ML runners only operate on `audit` and `syslog`.

Actors are treated as groups. By default:

- folders whose name contains `GPT` are treated as **AI**
- other actor folders are treated as **human**

This behavior is implemented in [`src/core/shared/actor_catalog.py`].

## Running the Agent Experiments

### Injecting a fault

Example:

```bash
python -m agent.scenarios.wordpress.wp_break4
```

By default, the scenario files are typically prepared to call `config(session)` in their `__main__` block. If you want to restore the system afterward, switch the file to call `fix(session)` instead.

### Recording a human troubleshooting session

For manual participants:

```bash
python -m agent.scenarios.nextcloud.human
python -m agent.scenarios.wordpress.human
```

These scripts:

1. connect to the server
2. store the current log offsets
3. wait until the participant has finished
4. extract only the new log entries

**Important:**

- Extracted logs are currently written to `agent/scenarios/wordpress/LOGS`, as defined in [`agent/utils.py`](agent/utils.py)
- They are not automatically integrated into the final `data/Nextcloud/...` or `data/WordPress/...` dataset structure
- In practice, collected logs need to be renamed and placed into the appropriate dataset folders

### Running the terminal LLM agent

```bash
python -m agent.runners.LLM_Agent
python -m agent.runners.LLM_Agent_WP
```

These agents:

- connect to the target via `pexpect`
- initialize log offsets
- interact with the shell
- optionally edit files through the Vim sub-agent
- extract newly generated logs at cleanup

**Important:**

- The current agents do not expose CLI arguments for the task description  
- The troubleshooting target is defined in `AgentConfig.problem_prompt`  
- Model configuration (e.g., model names, temperatures, recursion limits, and human-delay simulation) is specified directly in the source code  
- To reuse the same runner for a different fault scenario, the corresponding runner file can be adjusted accordingly  

### Running the browser agent

Nextcloud:

```bash
python -m agent.runners.browser_agent --playwright
```

WordPress:

```bash
python -m agent.runners.browser_agent_WP --playwright
```

Behavior:

- both browser runners start an interactive terminal prompt and wait for a free-text `Query:` instruction
- the Nextcloud browser agent always logs in using:
  - username `admin`
  - password `changeme`
- the WordPress browser agent logs in to `/wp-admin` only when needed using:
  - username `wpadmin`
  - password `changeme`
- the Nextcloud browser runner currently executes one query and exits
- the WordPress browser runner currently stays in a debug loop until `quit` or `exit`

Example query:

```text
Open wordpress.local, log in, and verify that the basic functionality of the site is working as expected.
```

## Machine Learning Experiment Runners

The main ML entry points are in `src/runners/ml/`.

All of these runners use **nested evaluation logic**:

- outer split: hold-out validation/test actor pairs
- inner search: select configuration using validation performance only
- final reporting: evaluate the selected configuration on the test split

The actor pair splits are generated in [`src/core/ml/val_test_combs.py`](src/core/ml/val_test_combs.py).

### Common concepts

- `--dataset`
  Selects `Nextcloud` or `WordPress`
  Legacy aliases also exist: `Data` = `Nextcloud`, `Data_WP` = `WordPress`
- `--limit_outer`
  Restricts the number of outer splits
- `--out_csv`
  Output CSV path for experiment results
- `--benchmark`
  Prints timing information for expensive steps

### 1. TF-IDF pipeline

Entry point:

```bash
python -m src.runners.ml.tfidf_360_nested --help
```

Purpose:

- sparse lexical baseline for log classification
- supports multiple classical models
- supports null-hypothesis label randomization

Arguments:

- `--dataset {Nextcloud,WordPress,Data,Data_WP}`
- `--model {dummy_most_frequent,dummy_stratified,svm,logreg,sgd_hinge,sgd_log,pa_like,ridge,mnb,cnb,bnb}`
- `--log_type {audit,syslog,nextcloud}`
- `--out_csv PATH`
- `--limit_outer INT`
- `--n_jobs INT`
- `--benchmark`
- `--randomize_actor_labels`
- `--assignment_idx INT`

Example:

```bash
python -m src.runners.ml.tfidf_360_nested \
  --dataset Nextcloud \
  --model svm \
  --log_type audit \
  --n_jobs 4 \
  --out_csv results/tfidf_nextcloud_audit_svm.csv
```

Important notes:

- for `WordPress`, `--log_type nextcloud` is not meaningful
- `--randomize_actor_labels` and `--assignment_idx` are used for null-hypothesis experiments

### 2. Inter-event time pipeline

Entry point:

```bash
python -m src.runners.ml.inter_times_360_nested --help
```

Purpose:

- classifies actors based only on time differences between events
- useful when studying behavioral timing rather than lexical content

Arguments:

- `--dataset {Nextcloud,WordPress,Data,Data_WP}`
- `--model {dummy_most_frequent,dummy_stratified,gnb,logreg,svm,sgd_hinge,sgd_log,ridge,knn}`
- `--log_type {audit,syslog,nextcloud}`
- `--metric {f1_macro,f1_weighted,accuracy,balanced_accuracy}`
- `--out_csv PATH`
- `--limit_outer INT`
- `--n_jobs INT`
- `--clip_max FLOAT`
- `--benchmark`

Example:

```bash
python -m src.runners.ml.inter_times_360_nested \
  --dataset Nextcloud \
  --model logreg \
  --log_type syslog \
  --clip_max 3600 \
  --n_jobs 4 \
  --out_csv results/inter_times_nextcloud_syslog_logreg.csv
```

### 3. CNN pipeline

Entry point:

```bash
python -m src.runners.ml.cnn_360_nested --help
```

Purpose:

- deep learning baseline over tokenized log windows

Arguments:

- `--dataset {Nextcloud,WordPress,Data,Data_WP}`
- `--out_csv PATH`
- `--metric {f1_macro,f1_weighted,accuracy,balanced_accuracy}`
- `--limit_outer INT`
- `--benchmark`

Example:

```bash
python -m src.runners.ml.cnn_360_nested \
  --dataset Nextcloud \
  --limit_outer 5 \
  --out_csv results/cnn_debug.csv
```

### 4. Transformer / BERT pipeline

Entry point:

```bash
python -m src.runners.ml.bert_360_nested --help
```

Purpose:

- transformer-based text classification over log windows

Arguments:

- `--dataset {Nextcloud,WordPress,Data,Data_WP}`
- `--out_csv PATH`
- `--metric {f1_macro,f1_weighted,accuracy,balanced_accuracy}`
- `--limit_outer INT`
- `--benchmark`

Example:

```bash
python -m src.runners.ml.bert_360_nested \
  --dataset WordPress \
  --limit_outer 3 \
  --out_csv results/bert_wordpress_debug.csv
```

### 5. Retrieval + LLM pipeline

Entry point:

```bash
python -m src.runners.ml.llm_360_nested --help
```

Purpose:

- retrieval-based pipeline with optional LLM fallback
- useful when exploring whether a semantic retrieval plus generative decision layer can distinguish actors

Arguments:

- `--dataset {Nextcloud,WordPress,Data,Data_WP}`
- `--out_csv PATH`
- `--metric {f1_macro,f1_weighted,accuracy,balanced_accuracy}`
- `--limit_outer INT`
- `--use_llm_fallback {0,1}`
- `--benchmark`

Example:

```bash
python -m src.runners.ml.llm_360_nested \
  --dataset Nextcloud \
  --use_llm_fallback 1 \
  --limit_outer 2 \
  --out_csv results/llm_nested_debug.csv
```

This runner requires `OPENAI_API_KEY` when the fallback is enabled.

## Statistical Experiment Runners

The statistical entry points are in `src/runners/stats/`.

These scripts evaluate whether AI-human separability appears in distributional properties of the logs, rather than only through supervised classification.

### 1. One-gram runner

Entry point:

```bash
python -m src.runners.stats.one_gram_runner --help
```

Modes:

- `single`: evaluate one concrete configuration
- `sweep`: evaluate many configurations automatically

Arguments:

- `--mode {single,sweep}`
- `--dataset {Nextcloud,WordPress}`
- `--assignment_mode {true,random_stratified,indexed_stratified}`
- `--assignment_idx INT`
- `--out_csv PATH`
- `--log_type {syslog,nextcloud,audit}`
- `--ngram_mode {char,word}` for `single`
- `--metric {js,l1}` for `single`

Example:

```bash
python -m src.runners.stats.one_gram_runner \
  --mode single \
  --dataset Nextcloud \
  --assignment_mode true \
  --log_type audit \
  --ngram_mode char \
  --metric js
```

### 2. Complexity metrics runner

Entry point:

```bash
python -m src.runners.stats.complexity_metrics_runner --help
```

Modes:

- `single`
- `sweep`

Arguments:

- `--mode {single,sweep}`
- `--dataset {Nextcloud,WordPress}`
- `--assignment_mode {true,random_stratified,indexed_stratified}`
- `--assignment_idx INT`
- `--out_csv PATH`
- `--log_type {syslog,nextcloud,audit}`
- `--window_size INT` for `single`
- `--stride INT` for `single`
- `--metric {gini,kurtosis,mad,entropy}` for `single`

Example:

```bash
python -m src.runners.stats.complexity_metrics_runner \
  --mode single \
  --dataset WordPress \
  --assignment_mode true \
  --log_type syslog \
  --window_size 5 \
  --stride 1 \
  --metric entropy
```

### 3. Focused audit row extractor

Entry point:

```bash
python -m src.runners.stats.audit_row_extractor_focused --help
```

Purpose:

- targeted exploratory analysis of audit record fields
- plots actor-wise distributions over selected audit attributes

Key arguments:

- `--dataset {Nextcloud,WordPress}`
- `--mode {execve,path,syscall,sockaddr}`
- `--distribution {conditional_distribution,presence_pattern,pair_distribution}`
- `--key1`, `--key2`
- `--given_key`, `--given_value`, `--target_key`
- `--top_k INT`
- `--n_humans INT`
- `--n_ais INT`
- `--save_path PATH`

### 4. Advanced audit timing analysis

Entry point:

```bash
python -m src.runners.stats.audit_advanced_time_focused --help
```

Purpose:

- groups audit lines into bundles and clusters
- analyzes inter-event timing and command-following timing
- mainly intended for detailed exploratory analysis rather than bulk pipeline execution

## Post-processing and Result Analysis

The `src/analysis/` directory contains utilities for interpreting the outputs produced by the runners.

### Null-hypothesis evaluation for ML results

```bash
python -m src.analysis.null_hypothes_eval \
  --metric test_f1_macro \
  --null-dir results/null_runs \
  --actual-csv results/tfidf_nextcloud_audit_svm.csv \
  --output results/null_boxplot.pdf \
  --larger-is-better
```

This script:

- loads many null-hypothesis CSVs
- computes the mean metric per CSV
- compares them to the observed run
- computes an empirical p-value (with `--larger-is-better`, higher scores are treated as better)
- generates a boxplot

### TF-IDF token attribution report

```bash
python -m src.analysis.tfidf_mark_token --help
```

Purpose:

- reconstruct one selected TF-IDF experiment row
- retrain the corresponding classifier on the same split
- generate an HTML report that highlights influential tokens or n-grams

This is useful when you want to inspect which lexical patterns drive attribution decisions in the TF-IDF baselines.

### Null-hypothesis evaluation for statistical runs

```bash
python -m src.analysis.stats_null_hypothes_eval \
  --metric silhouette \
  --null-dir results/stat_null_runs \
  --actual-csv results/statistics_real.csv \
  --output results/stat_null_boxplot.pdf
```
Purpose:

- selects the best-performing configuration per CSV according to the chosen metric
- builds a null distribution from these best values across permutations
- compares it to the observed best configuration
- computes an empirical p-value
- generates a boxplot of the null distribution with the observed value highlighted

### Ranking model result files by a metric

```bash
python -m src.analysis.rank_results_by_metric --help
```

Purpose:

- compare multiple result CSV files
- rank them by a chosen metric
- create a ranked boxplot

Example:

```bash
python -m src.analysis.rank_results_by_metric \
  --split-size 50 \
  --metric test_f1_macro \
  --results-dir results
```

### Ranking statistical configurations

```bash
python -m src.analysis.rank_statistic_configs_advanced --help
```

Purpose:

- rank statistical configurations by separation quality
- use scores such as:
  - mean separation
  - normalized mean separation
  - silhouette
  - Cliff's delta summaries

## Known Caveats

- The current agent connection helper uses the SSH alias `ssh arena_wp`.
- The log extraction helper operates on predefined log locations.
- The log output directory is currently defined directly in `agent/utils.py`.
- The terminal LLM agents define the troubleshooting task via `problem_prompt` in the source code rather than via CLI arguments.
- The terminal LLM agents invoke browser helpers via script names, making execution dependent on the working directory and Python executable configuration.
- Browser agents additionally rely on Playwright MCP and a Node.js installation.

These aspects do not affect the interpretation of the dataset provided in `data/`, but are relevant when reproducing the data collection process in a different environment.

## Suggested Starting Points

If you mainly want to work with the existing dataset:

1. Install dependencies.
2. Inspect the aggregated data under `data/Nextcloud/combine/ExperimentAggregated/` or `data/WordPress/combine/ExperimentAggregated/`.
3. Run a baseline such as `src.runners.ml.tfidf_360_nested`.
4. Run one of the statistical baselines such as `src.runners.stats.one_gram_runner`.
5. Use the scripts in `src/analysis/` to compare and interpret the outputs.

If you want to rerun the troubleshooting collection phase:

1. Adapt [`agent/utils.py`](agent/utils.py) to your SSH target and filesystem layout.
2. Prepare the corresponding Nextcloud or WordPress VM.
3. Apply one of the fault scenarios.
4. Execute the human or AI agent run.
5. Extract and organize the resulting logs into the expected `data/` layout.

## License and Attribution

This repository was created in the context of a master's thesis in cyber security. If you reuse the code or dataset structure, cite the thesis and clearly document any modifications to the experimental setup.
