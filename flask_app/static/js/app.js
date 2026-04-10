/**
 * WPARK Simulation — Main Application
 * Tabs: Neural Net | Networks leaderboard | Generation Report
 * High Speed mode skips simulation rendering during training.
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
        this.state           = null;
        this.autoRun         = false;
        this.autoRunInterval = null;
        this.animQueue       = [];
        this.drivingVehicles = new Set();
        this.fitnessHistory  = [];
        this.trainingActive  = false;
        this.trainingInterval= null;
        this._trainingTick   = 0;
        this.highSpeed       = false;   // skip sim steps during training
        this.popup           = null;
        this.activeTab       = 'nn';

        this.renderer = new ParkingPlanRenderer();
        this.animator = new VehicleAnimator(() =>
            document.getElementById('cpr-anim-layer'));

        this.animator.onNearMiss(() => {
            fetch('/api/near_miss', { method: 'POST' }).catch(() => {});
            const el = document.getElementById('stat-near-miss');
            if (el) el.textContent = (parseInt(el.textContent || '0') + 1).toString();
        });

        this.renderer.onBayClick((e, bayId, lvl) => this.showBayPopup(e, bayId, lvl));
        this.renderer.onVehicleClick((e, vid)    => this.showVehiclePopup(e, vid));

        this.config = { timeStep: 1.0, arrivalRate: 2.0, strategy: 'smart', simSpeed: 5 };
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
        document.getElementById('btn-high-speed').addEventListener('click',  () => this.toggleHighSpeed());
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
        document.getElementById('sim-speed').addEventListener('input', e => {
            this.config.simSpeed = parseInt(e.target.value);
            document.getElementById('sim-speed-value').textContent = this.config.simSpeed;
            // Restart auto-run interval at new rate if currently running
            if (this.autoRun) {
                clearInterval(this.autoRunInterval);
                this.autoRunInterval = setInterval(() => this.step(), this._stepIntervalMs());
            }
        });
        document.addEventListener('click', e => {
            if (this.popup.classList.contains('visible') &&
                !this.popup.contains(e.target) &&
                !e.target.closest('.bay') &&
                !e.target.closest('.sv-vehicle')) {
                this.closePopup();
            }
        });

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                this.activeTab = tab;
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(`tab-${tab}`).classList.add('active');
                // Refresh content immediately
                if (tab === 'networks' || tab === 'report') {
                    this.fetchGenerationReport();
                }
            });
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
            // During training use 5-min steps to match the evaluation granularity.
            const timeDelta = this.trainingActive ? 5.0 : this.config.timeStep;
            const res = await fetch('/api/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ time_delta: timeDelta })
            });
            const data = await res.json();
            // Backend reset the simulation for the new generation — clear all
            // client-side animation state so nothing stale plays.
            if (data.reset) {
                this.animQueue = [];
                this.drivingVehicles.clear();
                this.animator.clear();
            }
            // In high-speed training mode, skip animations but still update the map
            if (!(this.highSpeed && this.trainingActive) && data.animations?.length) {
                this.animQueue.push(...data.animations);
            }
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

    _stepIntervalMs() {
        return Math.round(1000 / this.config.simSpeed);
    }

    toggleAutoRun() {
        this.autoRun = !this.autoRun;
        const btn = document.getElementById('btn-auto');
        if (this.autoRun) {
            btn.textContent = '⏸ PAUSE'; btn.classList.add('active');
            this.autoRunInterval = setInterval(() => this.step(), this._stepIntervalMs());
        } else {
            btn.textContent = '▶ AUTO'; btn.classList.remove('active');
            clearInterval(this.autoRunInterval);
        }
    }

    // ── High speed mode ──────────────────────────────────────────
    toggleHighSpeed() {
        this.highSpeed = !this.highSpeed;
        const btn   = document.getElementById('btn-high-speed');
        const badge = document.getElementById('to-speed-badge');
        btn.classList.toggle('active', this.highSpeed);
        btn.textContent = this.highSpeed ? '⚡ FAST ON' : '⚡ FAST';
        if (badge) badge.style.display = this.highSpeed && this.trainingActive ? 'inline' : 'none';

        // Flush stale animations so vehicles that entered/exited during the
        // previous mode don't replay incorrectly in the new mode.
        this.animQueue = [];
        this.drivingVehicles.clear();
        this.animator.clear();

        // Sync the polling interval with the new speed setting.
        if (this.trainingActive) {
            clearInterval(this.trainingInterval);
            const pollMs = this.highSpeed ? 200 : 400;
            this.trainingInterval = setInterval(() => this.pollTraining(), pollMs);
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
                <svg width="16" height="16" style="flex-shrink:0">
                    <circle cx="8" cy="8" r="6" fill="${v.color}" opacity="0.9"/>
                    <circle cx="7" cy="7" r="2" fill="rgba(255,255,255,0.4)"/>
                </svg>
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
        }, 100);
    }

    playAnim(anim) {
        if (!this.state) return;
        if (this.highSpeed && this.trainingActive) return; // skip in high speed

        try {
            if (anim.type === 'enter') {
                const vehicle = anim.vehicle;
                const path = this.renderer.computeEntryPath(vehicle, anim.level, anim.bay_id, this.state.levels);
                if (!path) { console.warn('No entry path for', anim); return; }
                this.drivingVehicles.add(vehicle.id);
                this.animator.start(vehicle, path, () => { this.drivingVehicles.delete(vehicle.id); });
            }
            if (anim.type === 'exit') {
                const vehicle = anim.vehicle;
                const path = this.renderer.computeExitPath(vehicle, anim.level, anim.bay_id, this.state.levels);
                if (!path) { console.warn('No exit path for', anim); return; }
                this.drivingVehicles.delete(vehicle.id); // remove from driving set on exit
                this.animator.start(vehicle, path, () => {});
            }
        } catch(e) {
            console.error('playAnim error:', e, anim);
        }
    }

    // ── Popup system ─────────────────────────────────────────────
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
            const st  = this.state.current_time - vehicle.arrival_time;
            const pct = Math.min(100, (st / vehicle.estimated_stay_minutes) * 100);
            vBlock = `
                <div class="popup-divider"></div>
                <div class="popup-sub">PARKED VEHICLE</div>
                <div class="popup-row-c">
                    <svg width="16" height="16"><circle cx="8" cy="8" r="6" fill="${vehicle.color}"/><circle cx="7" cy="7" r="2" fill="rgba(255,255,255,0.4)"/></svg>
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
        const badges   = [
            v.requires_blue_badge   ? `<span class="acc-badge" style="background:var(--info)">♿ Blue Badge</span>` : '',
            v.requires_parent_child ? `<span class="acc-badge" style="background:#aa66cc">👶 P&amp;C</span>` : '',
            v.requires_ev_charging  ? `<span class="acc-badge" style="background:var(--success);color:#000">⚡ EV</span>` : ''
        ].filter(Boolean).join('');

        document.getElementById('popup-title').innerHTML =
            `<svg width="12" height="12" style="vertical-align:middle;margin-right:5px"><circle cx="6" cy="6" r="5" fill="${v.color}"/></svg>${v.id}`;
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

        layers.forEach((layer, li) => {
            if (li >= layers.length - 1) return;
            const nx = colX[li], nx2 = colX[li + 1];
            const show  = Math.min(layer.output_size,          maxShow[li]);
            const show2 = Math.min(layers[li+1].output_size,   maxShow[li+1]);
            const sp    = (H - 36) / (show  + 1);
            const sp2   = (H - 36) / (show2 + 1);
            const acts  = layer.activations || [];
            for (let n = 0; n < show; n++) {
                const y1  = 18 + sp * (n + 1);
                const act = Math.abs(acts[n] || 0);
                const op  = Math.max(0.02, Math.min(act * 0.20, 0.20));
                const stride = Math.max(1, Math.ceil(show2 / 4));
                for (let t = 0; t < show2; t += stride) {
                    const y2 = 18 + sp2 * (t + 1);
                    const cx = (nx + nx2) / 2;
                    const p  = this._svgEl('path');
                    p.setAttribute('d', `M${nx},${y1} C${cx},${y1} ${cx},${y2} ${nx2},${y2}`);
                    p.setAttribute('stroke', `rgba(255,136,0,${op})`);
                    p.setAttribute('stroke-width', '0.7');
                    p.setAttribute('fill', 'none');
                    svg.appendChild(p);
                }
            }
        });

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
                const bl  = act < 0 ? 255 : 0;
                if (abs > 0.45) {
                    const glow = this._svgEl('circle');
                    glow.setAttribute('cx', x); glow.setAttribute('cy', y); glow.setAttribute('r', 8);
                    glow.setAttribute('fill', `rgba(255,136,0,${abs * 0.14})`);
                    svg.appendChild(glow);
                }
                const c = this._svgEl('circle');
                c.setAttribute('cx', x); c.setAttribute('cy', y);
                c.setAttribute('r', li === 0 ? 5 : 4.5);
                c.setAttribute('fill',   `rgba(${r},${gb},${bl},${int})`);
                c.setAttribute('stroke', `rgba(255,136,0,0.35)`);
                c.setAttribute('stroke-width', '0.5');
                svg.appendChild(c);
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

    // ── Training ──────────────────────────────────────────────────
    async startTraining() {
        try {
            this.config.strategy = 'neural_network';
            document.getElementById('strategy-select').value = 'neural_network';
            await this.updateConfig();

            await fetch('/api/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    population_size: 30,
                    elite_count: 4,
                    fast_mode: this.highSpeed
                })
            });

            this.trainingActive  = true;
            this._trainingTick   = 0;
            this._lastEvalCount  = 0;
            // Speed up animations to match the faster sim cadence during training.
            this.animator.setSpeedScale(3.0);
            document.getElementById('btn-train-start').style.display = 'none';
            document.getElementById('btn-train-stop').style.display  = 'block';
            document.getElementById('mode-indicator').textContent    = 'TRAINING';
            document.getElementById('mode-indicator').classList.add('training');
            document.body.classList.add('is-training');

            const badge = document.getElementById('to-speed-badge');
            if (badge) badge.style.display = this.highSpeed ? 'inline' : 'none';

            // Poll more frequently in fast mode so UI stays responsive
            const pollMs = this.highSpeed ? 200 : 400;
            this.trainingInterval = setInterval(() => this.pollTraining(), pollMs);
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

            // Overlay panel
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
            document.getElementById('to-workers').textContent  = data.parallel_workers > 0 ? data.parallel_workers : '—';

            // Keep speed badge in sync with what backend reports
            const badge = document.getElementById('to-speed-badge');
            if (badge) badge.style.display = data.fast_mode ? 'inline' : 'none';

            this._trainingTick++;

            if (this._trainingTick % 3 === 0) this.updateNN();

            // Step the live sim once per newly-completed evaluation so the
            // animation advances at the same pace as the evaluation simulations.
            const prevEvalCount = this._lastEvalCount ?? 0;
            const newEvalCount  = ind + (data.generation * tot);   // monotonically rising
            const evalsDone     = newEvalCount - prevEvalCount;
            this._lastEvalCount = newEvalCount;

            if (evalsDone > 0) {
                // Cap at 3 steps per poll so we don't flood the backend.
                const stepsToRun = Math.min(evalsDone, 3);
                for (let i = 0; i < stepsToRun; i++) await this.step();
            } else {
                // Fallback: always step at least every 5 ticks so the map stays live.
                if (this._trainingTick % 5 === 0) await this.step();
            }

            if (data.fitness_history?.length) {
                this.fitnessHistory = data.fitness_history;
                this.drawFitnessGraph();
            }

            // Fetch generation report periodically
            if (this._trainingTick % 4 === 0) {
                this.fetchGenerationReport();
            }

            // Backend signalled it stopped (e.g. finished all generations)
            if (!data.active && this.trainingActive) this._stopTrainingUI();
        } catch(e) { console.error(e); }
    }

    async stopTraining() {
        // Immediately update UI so the button feels responsive
        this._stopTrainingUI();
        // Signal backend — fire-and-forget is fine; stop_event exits the loop fast now
        try {
            await fetch('/api/training/stop', { method: 'POST' });
        } catch(e) {}
    }

    _stopTrainingUI() {
        this.trainingActive = false;
        clearInterval(this.trainingInterval);
        this.trainingInterval = null;
        this.animator.setSpeedScale(1.0);   // restore normal animation speed

        document.getElementById('btn-train-start').style.display = 'block';
        document.getElementById('btn-train-stop').style.display  = 'none';
        document.getElementById('mode-indicator').textContent    = 'SIMULATION';
        document.getElementById('mode-indicator').classList.remove('training');
        document.body.classList.remove('is-training');

        const badge = document.getElementById('to-speed-badge');
        if (badge) badge.style.display = 'none';

        const prog = document.getElementById('train-progress');
        if (prog) prog.style.width = '0';

        // Fetch any reports that completed before we stopped
        this.fetchGenerationReport();
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
        ctx.strokeStyle = 'rgba(255,136,0,0.08)'; ctx.lineWidth = 1;
        for (let i = 0; i <= 3; i++) { const y = 4 + (H-8)*(i/3); ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
        ctx.fillStyle = 'rgba(255,136,0,0.06)';
        ctx.beginPath(); ctx.moveTo(0, H);
        hist.forEach((v,i) => ctx.lineTo(px(i), py(v)));
        ctx.lineTo(W, H); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = '#ff8800'; ctx.lineWidth = 1.5;
        ctx.beginPath();
        hist.forEach((v,i) => i===0 ? ctx.moveTo(px(i),py(v)) : ctx.lineTo(px(i),py(v)));
        ctx.stroke();
        ctx.fillStyle = 'rgba(255,204,0,0.9)';
        ctx.font = '9px Space Mono, monospace';
        ctx.fillText(hist[hist.length-1].toFixed(1), 3, 11);
    }

    // ── Generation Report ─────────────────────────────────────────
    async fetchGenerationReport() {
        try {
            const data = await (await fetch('/api/training/generation_report')).json();
            if (this.activeTab === 'networks' && data.latest) {
                this.renderNetworksTab(data.latest);
            }
            if (this.activeTab === 'report' && data.reports?.length) {
                this.renderReportTab(data.reports);
            }
        } catch(e) { console.error(e); }
    }

    renderNetworksTab(report) {
        const el = document.getElementById('networks-content');
        if (!el) return;
        el.innerHTML = `
            <div class="networks-header">
                <span>Generation ${report.generation}</span>
                <span class="networks-pop">${report.individuals.length} networks</span>
            </div>
            <div class="networks-meta">
                <span>Best: <strong>${report.best_fitness}</strong></span>
                <span>Avg: <strong>${report.avg_fitness}</strong></span>
            </div>
            <div class="networks-table">
                ${report.individuals.map(ind => `
                    <div class="network-row ${ind.rank === 1 ? 'network-row-best' : ''}">
                        <div class="net-rank">#${ind.rank}</div>
                        <div class="net-score-wrap">
                            <div class="net-score-bar">
                                <div class="net-score-fill" style="width:${ind.score}%;background:${this._scoreColor(ind.score)}"></div>
                            </div>
                            <span class="net-score-val">${ind.score}<span class="net-score-unit">/100</span></span>
                        </div>
                        <div class="net-meta">
                            <span class="net-fitness">${ind.fitness.toFixed(0)}</span>
                            <span class="net-born">G${ind.generation_born}</span>
                        </div>
                    </div>
                `).join('')}
            </div>`;
    }

    renderReportTab(reports) {
        const el = document.getElementById('report-content');
        if (!el) return;
        el.innerHTML = [...reports].reverse().map(r => {
            const topScore = r.individuals[0]?.score ?? 0;
            return `
            <div class="report-gen">
                <div class="report-gen-header">
                    <span class="report-gen-num">GEN ${r.generation}</span>
                    <span class="report-gen-score" style="color:${this._scoreColor(topScore)}">${topScore}/100</span>
                    <span class="report-gen-avg">avg ${r.avg_fitness}</span>
                </div>
                <div class="report-gen-bar">
                    <div class="report-gen-fill" style="width:${topScore}%;background:${this._scoreColor(topScore)}"></div>
                </div>
                <div class="report-gen-grid">
                    ${r.individuals.slice(0, 6).map(ind => `
                        <div class="rg-cell" title="Fitness: ${ind.fitness} | Born: G${ind.generation_born}">
                            <div class="rg-bar" style="height:${ind.score}%;background:${this._scoreColor(ind.score)}"></div>
                            <div class="rg-rank">#${ind.rank}</div>
                        </div>
                    `).join('')}
                </div>
                <div class="report-gen-details">
                    Top: ${r.individuals.slice(0,3).map(i => `<span class="rg-pill" style="background:${this._scoreColor(i.score)}">${i.score}</span>`).join(' ')}
                    &nbsp;·&nbsp; ${r.individuals.length} networks evaluated
                </div>
            </div>`;
        }).join('');
    }

    _scoreColor(score) {
        if (score >= 80) return '#44ff66';
        if (score >= 60) return '#aaee22';
        if (score >= 40) return '#ff8800';
        if (score >= 20) return '#ff6600';
        return '#ff4444';
    }
}

let simulation;
document.addEventListener('DOMContentLoaded', () => { simulation = new WPARKSimulation(); });
