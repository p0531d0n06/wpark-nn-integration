/**
 * WPARK Simulation — Main Application
 * Uses ParkingPlanRenderer + VehicleAnimator from carParkRenderer.js
 */

const PURPOSE_COLORS = {
    emergency:     '#ff3333',
    medical:       '#5599ff',
    quick_errand:  '#ffdd00',
    shopping:      '#ff8800',
    dining:        '#ffaa22',
    entertainment: '#ff6622',
    commute:       '#888888'
};

const NN_INPUT_LABELS = [
    'size_c','size_s','size_l','size_x',
    'ev','blue_badge','p&c',
    'mob_0','mob_1','mob_2','mob_3','mob_4',
    'priority','stay','purpose',
    'occ_-1','occ_0','occ_1',
    'queue','sin_t','cos_t','day',
    'cp_occ','avail','occ_n'
];

class WPARKSimulation {
    constructor() {
        this.state         = null;
        this.autoRun       = false;
        this.autoRunInterval = null;
        this.animQueue     = [];        // pending animation events
        this.drivingVehicles = new Set();
        this.fitnessHistory = [];
        this.trainingActive = false;
        this.trainingInterval = null;
        this._trainingTick  = 0;
        this.popup = null;

        this.renderer = new ParkingPlanRenderer();
        this.animator = new VehicleAnimator(() =>
            document.getElementById('cpr-anim-layer'));

        this.animator.onNearMiss(() => {
            fetch('/api/near_miss', { method: 'POST' }).catch(() => {});
            const el = document.getElementById('stat-near-miss');
            if (el) el.textContent = (parseInt(el.textContent || '0') + 1).toString();
        });

        this.renderer.onBayClick((e, bayId, lvl)  => this.showBayPopup(e, bayId, lvl));
        this.renderer.onVehicleClick((e, vid)      => this.showVehiclePopup(e, vid));

        this.config = { timeStep: 1.0, arrivalRate: 2.0, strategy: 'smart' };
        this.init();
    }

    async init() {
        this.popup = document.getElementById('info-popup');
        this.bindEvents();
        await this.loadStrategies();
        await this.fetchState();
        this.render();
        this.startAnimLoop();
        this.updateNN();
    }

    bindEvents() {
        document.getElementById('btn-step').addEventListener('click',  () => this.step());
        document.getElementById('btn-reset').addEventListener('click', () => this.reset());
        document.getElementById('btn-auto').addEventListener('click',  () => this.toggleAutoRun());
        document.getElementById('btn-train-start').addEventListener('click', () => this.startTraining());
        document.getElementById('btn-train-stop').addEventListener('click',  () => this.stopTraining());
        document.getElementById('popup-close').addEventListener('click', () => this.closePopup());

        document.getElementById('time-step').addEventListener('input', e => {
            this.config.timeStep = parseFloat(e.target.value);
            document.getElementById('time-step-value').textContent = this.config.timeStep.toFixed(1);
        });
        document.getElementById('arrival-rate').addEventListener('input', e => {
            this.config.arrivalRate = parseFloat(e.target.value);
            document.getElementById('arrival-rate-value').textContent = this.config.arrivalRate.toFixed(1);
            this.updateConfig();
        });
        document.getElementById('strategy-select').addEventListener('change', e => {
            this.config.strategy = e.target.value;
            this.updateConfig();
        });
        document.addEventListener('click', e => {
            if (this.popup.classList.contains('visible') &&
                !this.popup.contains(e.target) &&
                !e.target.closest('.bay') &&
                !e.target.closest('.sv-vehicle')) {
                this.closePopup();
            }
        });
    }

    async loadStrategies() {
        try {
            const strategies = await (await fetch('/api/strategies')).json();
            const sel = document.getElementById('strategy-select');
            sel.innerHTML = strategies.map(s =>
                `<option value="${s}" ${s === this.config.strategy ? 'selected' : ''}>${s.replace(/_/g,' ').toUpperCase()}</option>`
            ).join('');
        } catch(e) { console.error(e); }
    }

    async fetchState() {
        try {
            const data = await (await fetch('/api/state')).json();
            this.state = data;
            if (data.animations?.length) this.animQueue.push(...data.animations);
        } catch(e) { console.error(e); }
    }

    // ── Simulation step ──────────────────────────────────────────
    async step() {
        try {
            const res = await fetch('/api/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ time_delta: this.config.timeStep })
            });
            const data = await res.json();
            if (data.animations?.length) this.animQueue.push(...data.animations);
            await this.fetchState();
            this.render();
            this.updateNN();
        } catch(e) { console.error(e); }
    }

    async reset() {
        try {
            await fetch('/api/reset', { method: 'POST' });
            this.drivingVehicles.clear();
            this.animator.clear();
            this.animQueue = [];
            await this.fetchState();
            this.render();
        } catch(e) { console.error(e); }
    }

    async updateConfig() {
        try {
            await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    arrival_rate:        this.config.arrivalRate,
                    assignment_strategy: this.config.strategy
                })
            });
        } catch(e) {}
    }

    toggleAutoRun() {
        this.autoRun = !this.autoRun;
        const btn = document.getElementById('btn-auto');
        if (this.autoRun) {
            btn.textContent = '⏸ PAUSE'; btn.classList.add('active');
            this.autoRunInterval = setInterval(() => this.step(), 350);
        } else {
            btn.textContent = '▶ AUTO'; btn.classList.remove('active');
            clearInterval(this.autoRunInterval);
        }
    }

    // ── Render ───────────────────────────────────────────────────
    render() {
        if (!this.state) return;
        this.renderHeader();
        this.renderStats();
        this.renderQueue();

        const container = document.getElementById('carpark-container');
        this.renderer.render(container, this.state.levels, this.state, this.drivingVehicles);
    }

    renderHeader() {
        const t = this.state.current_time;
        const h = Math.floor(t / 60), m = Math.floor(t % 60);
        document.getElementById('sim-time').textContent =
            `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`;
    }

    renderStats() {
        const s = this.state.stats;
        const t = this.state.traffic || {};
        document.getElementById('stat-occupancy').textContent  = `${(this.state.occupancy_rate * 100).toFixed(0)}%`;
        document.getElementById('stat-arrivals').textContent   = s.total_arrivals;
        document.getElementById('stat-departures').textContent = s.total_departures;
        document.getElementById('stat-queue').textContent      = this.state.waiting_queue.length;
        document.getElementById('stat-denied').textContent     = s.total_denied;
        document.getElementById('stat-avg-stay').textContent   = `${s.avg_stay_duration.toFixed(0)}m`;
        // Traffic
        const entryQEl = document.getElementById('stat-entry-queue');
        if (entryQEl) entryQEl.textContent = t.entry_queue_length ?? 0;
        const transitEl = document.getElementById('stat-in-transit');
        if (transitEl) transitEl.textContent = t.in_transit_count ?? 0;
        const congEl = document.getElementById('stat-congestion');
        if (congEl) congEl.textContent = t.congestion_events ?? 0;
    }

    renderQueue() {
        const el = document.getElementById('queue-container');
        const q  = this.state.waiting_queue;
        if (!q.length) { el.innerHTML = '<div class="queue-empty">No vehicles waiting</div>'; return; }
        el.innerHTML = q.slice(0, 5).map(v => `
            <div class="queue-item">
                <div class="queue-car" style="background:${v.color}"></div>
                <div class="queue-info">
                    <div class="queue-id">${v.id}</div>
                    <div class="queue-detail">${v.purpose.replace(/_/g,' ')} → ${v.target_shop || 'any'}</div>
                </div>
            </div>
        `).join('');
        if (q.length > 5) el.innerHTML += `<div class="queue-more">+${q.length - 5} more</div>`;
    }

    // ── Animation loop ───────────────────────────────────────────
    startAnimLoop() {
        setInterval(() => {
            if (!this.animQueue.length) return;
            const batch = this.animQueue.splice(0, this.animQueue.length);
            batch.forEach(a => this.playAnim(a));
        }, 120);
    }

    playAnim(anim) {
        if (!this.state) return;

        if (anim.type === 'enter') {
            const vehicle = anim.vehicle;
            const path = this.renderer.computeEntryPath(
                vehicle, anim.level, anim.bay_id, this.state.levels
            );
            if (!path) return;

            this.drivingVehicles.add(vehicle.id);
            this.animator.start(vehicle, path, () => {
                this.drivingVehicles.delete(vehicle.id);
            });
        }

        if (anim.type === 'exit') {
            const vehicle = anim.vehicle;
            const path = this.renderer.computeExitPath(
                vehicle, anim.level, anim.bay_id, this.state.levels
            );
            if (!path) return;
            this.animator.start(vehicle, path, () => {});
        }
    }

    // ── Popup system ────────────────────────────────────────────
    showBayPopup(event, bayId, levelNum) {
        if (!this.state) return;
        const level = this.state.levels.find(l => l.level_number === levelNum);
        if (!level) return;
        const bay = level.bays.find(b => b.id === bayId);
        if (!bay) return;

        const vehicle = bay.occupied_by ? this.state.active_vehicles[bay.occupied_by] : null;

        const typeIcon  = { standard:'🅿', blue_badge:'♿', parent_child:'👶', ev:'⚡' };
        const typeColor = { standard:'var(--text-muted)', blue_badge:'var(--info)',
                            parent_child:'#aa66cc', ev:'var(--success)' };

        let vBlock = '';
        if (vehicle) {
            const st = this.state.current_time - vehicle.arrival_time;
            const pct = Math.min(100, (st / vehicle.estimated_stay_minutes) * 100);
            vBlock = `
                <div class="popup-divider"></div>
                <div class="popup-sub">PARKED VEHICLE</div>
                <div class="popup-row-c">
                    <span class="popup-swatch" style="background:${vehicle.color}"></span>
                    <span class="popup-hl">${vehicle.id}</span>
                </div>
                <div class="popup-kv"><span class="pk">Purpose</span>
                    <span class="popup-badge pb-purpose" style="background:${PURPOSE_COLORS[vehicle.purpose]||'#888'}">${vehicle.purpose.replace(/_/g,' ')}</span></div>
                <div class="popup-kv"><span class="pk">Target</span><span class="pv hl">${vehicle.target_shop||'Any'}</span></div>
                <div class="popup-kv"><span class="pk">Stay</span><span class="pv">${st.toFixed(0)}/${vehicle.estimated_stay_minutes.toFixed(0)} min</span></div>
                <div class="popup-prog"><div class="popup-prog-fill" style="width:${pct}%;background:${vehicle.color}"></div></div>`;
        }

        document.getElementById('popup-title').textContent = `${typeIcon[bay.bay_type]||'🅿'} Bay ${bay.row}${bay.position}`;
        document.getElementById('popup-body').innerHTML = `
            <div class="popup-type-lbl" style="color:${typeColor[bay.bay_type]}">${typeIcon[bay.bay_type]} ${bay.bay_type.replace(/_/g,' ').toUpperCase()}</div>
            <div class="popup-status-pill ${bay.status}">${bay.status.toUpperCase()}</div>
            <div class="popup-kv"><span class="pk">Bay ID</span><span class="pv">${bay.id}</span></div>
            <div class="popup-kv"><span class="pk">Shop Gate</span><span class="pv">${bay.distance_to_lift.toFixed(0)}u</span></div>
            <div class="popup-kv"><span class="pk">Exit Dist</span><span class="pv">${bay.distance_to_exit.toFixed(0)}u</span></div>
            ${vBlock}`;

        this._posPopup(event);
        this.popup.classList.add('visible');
    }

    showVehiclePopup(event, vehicleId) {
        if (!this.state) return;
        const v = this.state.active_vehicles[vehicleId];
        if (!v) return;

        const stayTime = this.state.current_time - v.arrival_time;
        const pct      = Math.min(100, (stayTime / v.estimated_stay_minutes) * 100);
        const leftMins = Math.max(0, v.estimated_stay_minutes - stayTime);
        const sizeI    = { compact:'🚗', standard:'🚙', large:'🚐', oversized:'🚛' };

        const badges = [
            v.requires_blue_badge    ? `<span class="acc-badge" style="background:var(--info)">♿ Blue Badge</span>` : '',
            v.requires_parent_child  ? `<span class="acc-badge" style="background:#aa66cc">👶 P&amp;C</span>` : '',
            v.requires_ev_charging   ? `<span class="acc-badge" style="background:var(--success);color:#000">⚡ EV</span>` : ''
        ].filter(Boolean).join('');

        document.getElementById('popup-title').innerHTML =
            `<span style="display:inline-block;width:10px;height:10px;background:${v.color};border-radius:2px;margin-right:5px"></span>${v.id}`;

        document.getElementById('popup-body').innerHTML = `
            <div class="popup-header-row">
                <span class="popup-badge pb-purpose" style="background:${PURPOSE_COLORS[v.purpose]||'#888'}">${v.purpose.replace(/_/g,' ')}</span>
                <span class="popup-badge pb-size">${sizeI[v.size]||'🚗'} ${v.size}</span>
            </div>
            <div class="popup-kv"><span class="pk">Target Shop</span><span class="pv hl">${v.target_shop||'Any'}</span></div>
            <div class="popup-kv"><span class="pk">Assigned Bay</span><span class="pv">${v.assigned_bay_id||'—'}</span></div>
            <div class="popup-kv"><span class="pk">Level</span><span class="pv">Level ${v.assigned_level}</span></div>
            <div class="popup-kv"><span class="pk">Mobility</span><span class="pv">${v.mobility.replace(/_/g,' ')}</span></div>
            <div class="popup-divider"></div>
            <div class="popup-sub">STAY DURATION</div>
            <div class="popup-stay-row">
                <span>${stayTime.toFixed(0)} min elapsed</span>
                <span>${leftMins.toFixed(0)} min left</span>
            </div>
            <div class="popup-prog"><div class="popup-prog-fill" style="width:${pct}%;background:${v.color}"></div></div>
            <div class="popup-kv" style="margin-top:8px">
                <span class="pk">Priority</span>
                <span class="pv">${(v.priority * 10).toFixed(1)} / 10</span>
            </div>
            ${badges ? `<div class="popup-divider"></div><div class="popup-badges-row">${badges}</div>` : ''}`;

        this._posPopup(event);
        this.popup.classList.add('visible');
    }

    _posPopup(event) {
        const pw = 250, ph = 340;
        let x = event.clientX + 14, y = event.clientY + 14;
        if (x + pw > window.innerWidth  - 8) x = event.clientX - pw - 14;
        if (y + ph > window.innerHeight - 8) y = event.clientY - ph - 14;
        this.popup.style.left = x + 'px';
        this.popup.style.top  = y + 'px';
    }

    closePopup() { this.popup.classList.remove('visible'); }

    // ── Neural Network Visualisation ─────────────────────────────
    async updateNN() {
        try {
            const data = await (await fetch('/api/neural_network')).json();

            document.getElementById('nn-params').textContent     = data.total_params.toLocaleString();
            const conf = data.prediction_info?.confidence;
            document.getElementById('nn-confidence').textContent = conf != null ? `${(conf * 100).toFixed(1)}%` : '—';

            this.renderNNDiagram(data);
            this.renderNNPrediction(data.prediction_info);
            this.renderNNActivations(data.layers);
        } catch(e) { console.error(e); }
    }

    renderNNDiagram(data) {
        const svg = document.getElementById('nn-svg');
        svg.innerHTML = '';
        const W = 280, H = 240;
        const layers = data.layers;
        if (!layers?.length) return;

        const colX    = [66, 134, 202, 262];
        const maxShow = [14,  12,  10,   8];

        // Connections first (behind nodes)
        layers.forEach((layer, li) => {
            if (li >= layers.length - 1) return;
            const nx = colX[li], nx2 = colX[li + 1];
            const show  = Math.min(layer.output_size,            maxShow[li]);
            const show2 = Math.min(layers[li + 1].output_size,   maxShow[li + 1]);
            const sp    = (H - 36) / (show  + 1);
            const sp2   = (H - 36) / (show2 + 1);
            const acts  = layer.activations || [];

            for (let n = 0; n < show; n++) {
                const y1  = 18 + sp * (n + 1);
                const act = Math.abs(acts[n] || 0);
                const op  = Math.max(0.025, Math.min(act * 0.22, 0.22));
                const stride = Math.max(1, Math.ceil(show2 / 4));
                for (let t = 0; t < show2; t += stride) {
                    const y2 = 18 + sp2 * (t + 1);
                    const cx = (nx + nx2) / 2;
                    const p  = this._svgEl('path');
                    p.setAttribute('d', `M${nx},${y1} C${cx},${y1} ${cx},${y2} ${nx2},${y2}`);
                    p.setAttribute('stroke',       `rgba(255,136,0,${op})`);
                    p.setAttribute('stroke-width', '0.7');
                    p.setAttribute('fill',         'none');
                    svg.appendChild(p);
                }
            }
        });

        // Neurons
        layers.forEach((layer, li) => {
            const x    = colX[li];
            const tot  = layer.output_size;
            const show = Math.min(tot, maxShow[li]);
            const sp   = (H - 36) / (show + 1);
            const acts = layer.activations || [];

            for (let n = 0; n < show; n++) {
                const y   = 18 + sp * (n + 1);
                const act = acts[n] || 0;
                const abs = Math.abs(act);
                const int = 0.25 + Math.min(abs, 1) * 0.75;
                const r   = act >= 0 ? 255 : Math.floor(80 * (1 - abs));
                const gb  = Math.floor(80 + abs * 56);
                const bl  = act <  0 ? 255 : 0;

                if (abs > 0.45) {
                    const glow = this._svgEl('circle');
                    glow.setAttribute('cx', x); glow.setAttribute('cy', y); glow.setAttribute('r', 8);
                    glow.setAttribute('fill', `rgba(255,136,0,${abs * 0.14})`);
                    svg.appendChild(glow);
                }
                const c = this._svgEl('circle');
                c.setAttribute('cx', x); c.setAttribute('cy', y);
                c.setAttribute('r',  li === 0 ? 5 : 4.5);
                c.setAttribute('fill',   `rgba(${r},${gb},${bl},${int})`);
                c.setAttribute('stroke', `rgba(255,136,0,0.35)`);
                c.setAttribute('stroke-width', '0.5');
                svg.appendChild(c);

                // Input labels
                if (li === 0 && n < NN_INPUT_LABELS.length) {
                    const lbl = this._svgEl('text');
                    lbl.setAttribute('x', x - 8); lbl.setAttribute('y', y + 3);
                    lbl.setAttribute('text-anchor', 'end');
                    lbl.setAttribute('fill', 'rgba(255,136,0,0.5)');
                    lbl.setAttribute('font-size', '6.5');
                    lbl.setAttribute('font-family', 'Space Mono,monospace');
                    lbl.textContent = NN_INPUT_LABELS[n];
                    svg.appendChild(lbl);
                }
            }

            if (tot > show) {
                const d = this._svgEl('text');
                d.setAttribute('x', x); d.setAttribute('y', 18 + sp * (show + 0.7));
                d.setAttribute('text-anchor', 'middle');
                d.setAttribute('fill', 'rgba(255,136,0,0.3)');
                d.setAttribute('font-size', '9');
                d.textContent = '···';
                svg.appendChild(d);
            }

            const lbl = this._svgEl('text');
            lbl.setAttribute('x', x); lbl.setAttribute('y', H - 4);
            lbl.setAttribute('text-anchor', 'middle');
            lbl.setAttribute('fill', 'rgba(255,136,0,0.6)');
            lbl.setAttribute('font-size', '7');
            lbl.setAttribute('font-family', 'Space Mono,monospace');
            lbl.textContent = `${layer.name.replace('Hidden ','H')}(${tot})`;
            svg.appendChild(lbl);
        });
    }

    renderNNPrediction(info) {
        if (!info?.vehicle_id) return;
        document.getElementById('nn-pred-vehicle').textContent = info.vehicle_id || '—';
        document.getElementById('nn-pred-bays').textContent    = info.available_count ?? '—';
        document.getElementById('nn-pred-bay').textContent     = info.selected_bay || '—';

        const conf = info.confidence || 0;
        document.getElementById('nn-pred-conf-val').textContent = `${(conf * 100).toFixed(1)}%`;
        const fill = document.getElementById('nn-conf-fill');
        fill.style.width      = `${conf * 100}%`;
        fill.style.background = conf > 0.7 ? 'var(--success)' : conf > 0.4 ? 'var(--accent-primary)' : 'var(--danger)';
    }

    renderNNActivations(layers) {
        const el = document.getElementById('nn-activations');
        if (!layers) return;
        el.innerHTML = layers
            .filter(l => l.activations?.length > 0)
            .map(layer => {
                const acts   = layer.activations.slice(0, 32);
                const maxAbs = Math.max(...acts.map(Math.abs), 0.001);
                const avgAbs = acts.reduce((a, b) => a + Math.abs(b), 0) / acts.length;
                const pct    = Math.min(100, (avgAbs / maxAbs) * 100);

                const sparks = acts.slice(0, 26).map(a => {
                    const n = Math.abs(a) / maxAbs;
                    const h = Math.max(2, n * 16);
                    const col = a >= 0 ? `rgba(255,136,0,${0.3 + n * 0.7})` : `rgba(68,136,255,${0.3 + n * 0.7})`;
                    return `<div class="act-spark" style="height:${h}px;background:${col}"></div>`;
                }).join('');

                return `
                <div class="act-row">
                    <div class="act-header">
                        <span class="act-name">${layer.name}</span>
                        <span class="act-avg">${avgAbs.toFixed(3)}</span>
                    </div>
                    <div class="act-bar-wrap"><div class="act-bar-fill" style="width:${pct}%"></div></div>
                    <div class="act-sparkline">${sparks}</div>
                </div>`;
            }).join('');
    }

    _svgEl(tag) { return document.createElementNS('http://www.w3.org/2000/svg', tag); }

    // ── Training ─────────────────────────────────────────────────
    async startTraining() {
        try {
            this.config.strategy = 'neural_network';
            document.getElementById('strategy-select').value = 'neural_network';
            await this.updateConfig();

            await fetch('/api/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ population_size: 20 })
            });

            this.trainingActive  = true;
            this._trainingTick   = 0;
            document.getElementById('btn-train-start').style.display = 'none';
            document.getElementById('btn-train-stop').style.display  = 'block';
            document.getElementById('mode-indicator').textContent = 'TRAINING';
            document.getElementById('mode-indicator').classList.add('training');
            document.body.classList.add('is-training');

            this.trainingInterval = setInterval(() => this.pollTraining(), 400);
        } catch(e) { console.error(e); }
    }

    async pollTraining() {
        if (!this.trainingActive) return;
        try {
            const data = await (await fetch('/api/training/status')).json();

            // Sidebar mini-stats
            document.getElementById('train-gen').textContent     = data.generation;
            document.getElementById('train-fitness').textContent = data.best_fitness.toFixed(1);
            const sideProg = document.getElementById('train-progress');
            if (sideProg && data.population_size > 0) {
                sideProg.style.width = Math.round((data.current_individual / data.population_size) * 100) + '%';
            }

            // Training overlay panel
            const ind = data.current_individual ?? 0;
            const tot = data.population_size   ?? 0;
            const pct = tot > 0 ? Math.round((ind / tot) * 100) : 0;

            document.getElementById('to-gen').textContent      = data.generation;
            document.getElementById('to-best').textContent     = data.best_fitness.toFixed(1);
            document.getElementById('to-avg').textContent      = data.stats?.avg       != null ? data.stats.avg.toFixed(1)       : '—';
            document.getElementById('to-div').textContent      = data.stats?.diversity != null ? data.stats.diversity.toFixed(3) : '—';
            document.getElementById('to-eval').textContent     = tot ? `${ind} / ${tot}` : '—';
            document.getElementById('to-prog').style.width     = pct + '%';
            document.getElementById('to-prog-pct').textContent = pct;
            const workers = data.parallel_workers;
            document.getElementById('to-workers').textContent  = workers > 0 ? workers : '—';

            this._trainingTick++;
            if (this._trainingTick % 3 === 0) this.updateNN();
            if (this._trainingTick % 5 === 0) await this.step();   // slower sim steps during training

            if (data.fitness_history?.length) {
                this.fitnessHistory = data.fitness_history;
                this.drawFitnessGraph();
            }

            if (!data.active && this.trainingActive) this._stopTrainingUI();
        } catch(e) { console.error(e); }
    }

    async stopTraining() {
        this._stopTrainingUI();
        fetch('/api/training/stop', { method: 'POST' }).catch(() => {});
    }

    _stopTrainingUI() {
        this.trainingActive = false;
        clearInterval(this.trainingInterval);
        this.trainingInterval = null;

        document.getElementById('btn-train-start').style.display = 'block';
        document.getElementById('btn-train-stop').style.display  = 'none';
        document.getElementById('mode-indicator').textContent = 'SIMULATION';
        document.getElementById('mode-indicator').classList.remove('training');
        document.body.classList.remove('is-training');

        const prog = document.getElementById('train-progress');
        if (prog) prog.style.width = '0';
    }

    drawFitnessGraph() {
        const canvas = document.getElementById('fitness-canvas');
        const ctx    = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;

        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, W, H);

        const hist = this.fitnessHistory;
        if (hist.length < 2) return;

        const mx = Math.max(...hist), mn = Math.min(...hist), rng = mx - mn || 1;
        const px = i => (i / (hist.length - 1)) * W;
        const py = v => 4 + (H - 8) * (1 - (v - mn) / rng);

        // Grid
        ctx.strokeStyle = 'rgba(255,136,0,0.08)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 3; i++) { const y = 4 + (H - 8) * (i / 3); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

        // Fill
        ctx.fillStyle = 'rgba(255,136,0,0.06)';
        ctx.beginPath(); ctx.moveTo(0, H);
        hist.forEach((v, i) => ctx.lineTo(px(i), py(v)));
        ctx.lineTo(W, H); ctx.closePath(); ctx.fill();

        // Line
        ctx.strokeStyle = '#ff8800'; ctx.lineWidth = 1.5;
        ctx.beginPath();
        hist.forEach((v, i) => i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)));
        ctx.stroke();

        // Latest value
        ctx.fillStyle = 'rgba(255,204,0,0.9)';
        ctx.font = '9px Space Mono, monospace';
        ctx.fillText(hist[hist.length - 1].toFixed(1), 3, 11);
    }
}

let simulation;
document.addEventListener('DOMContentLoaded', () => { simulation = new WPARKSimulation(); });
