/**
 * WPARK – SVG Parking Plan Renderer
 * Realistic overhead car-park layout with smooth breathing-dot vehicles.
 */

// ── Layout constants ──────────────────────────────────────────────────
const CPR = {
    SVG_W:       900,
    MARGIN_L:    64,    // left margin – ramp labels
    MARGIN_R:    46,    // right margin – entry / exit
    BAY_H:       42,    // bay depth (top-down)
    AISLE_H:     46,    // driving aisle
    LABEL_H:     26,    // floor-name strip
    CONNECTOR_H: 60,    // ramp section between floors
    PAD_BOTTOM:  10,
};

CPR.USABLE_W = CPR.SVG_W - CPR.MARGIN_L - CPR.MARGIN_R;  // 790 px

// Row Y positions relative to floor-group origin
//   LABEL | A | AISLE1 | B | C | AISLE2 | D | PAD
CPR.ROW_Y = {
    A: CPR.LABEL_H,
    B: CPR.LABEL_H + CPR.BAY_H + CPR.AISLE_H,
    C: CPR.LABEL_H + CPR.BAY_H + CPR.AISLE_H + CPR.BAY_H,
    D: CPR.LABEL_H + CPR.BAY_H + CPR.AISLE_H + CPR.BAY_H + CPR.BAY_H + CPR.AISLE_H,
};

CPR.FLOOR_H   = CPR.ROW_Y.D + CPR.BAY_H + CPR.PAD_BOTTOM;

CPR.AISLE1_Y  = CPR.ROW_Y.A + CPR.BAY_H;
CPR.AISLE1_CY = CPR.AISLE1_Y + CPR.AISLE_H / 2;
CPR.AISLE2_Y  = CPR.ROW_Y.C + CPR.BAY_H;
CPR.AISLE2_CY = CPR.AISLE2_Y + CPR.AISLE_H / 2;

CPR.ROW_FACES_DOWN = { A: true,  B: false, C: true,  D: false };
CPR.ROW_AISLE      = { A: 1,     B: 1,     C: 2,     D: 2    };

// Bay colours
const BAY_COLOURS = {
    standard:     { fill: '#0b1a0b', stroke: '#264020' },
    blue_badge:   { fill: '#050f1e', stroke: '#1a4080' },
    parent_child: { fill: '#0d0616', stroke: '#5e1f99' },
    ev:           { fill: '#041406', stroke: '#1a5e1a' },
    occupied:     { fill: '#190700', stroke: '#7a3010' },
    reserved:     { fill: '#11100a', stroke: '#604e00' },
};

// ── ParkingPlanRenderer ──────────────────────────────────────────────
class ParkingPlanRenderer {
    constructor() {
        this.NS = 'http://www.w3.org/2000/svg';
        this.levelOffsets = {};
        this.svg = null;
        this.animLayer = null;
        this._callbacks = { bay: null, vehicle: null };
    }

    onBayClick(fn)     { this._callbacks.bay     = fn; }
    onVehicleClick(fn) { this._callbacks.vehicle = fn; }

    // ── Main render ───────────────────────────────────────────────
    render(container, levels, state, drivingVehicles = new Set()) {
        const sorted = [...levels].sort((a, b) => b.level_number - a.level_number);

        this.levelOffsets = {};
        let yOff = 0;
        sorted.forEach((lv, i) => {
            this.levelOffsets[lv.level_number] = yOff;
            yOff += CPR.FLOOR_H + (i < sorted.length - 1 ? CPR.CONNECTOR_H : 0);
        });
        const totalH = yOff;

        const existingAnimLayer = document.getElementById('cpr-anim-layer');

        const svg = this._el('svg');
        svg.setAttribute('id', 'carpark-svg');
        svg.setAttribute('viewBox', `0 0 ${CPR.SVG_W} ${totalH}`);
        svg.setAttribute('width', '100%');
        svg.setAttribute('overflow', 'visible'); // allow animated cars to show briefly outside viewport
        svg.style.display   = 'block';
        svg.style.minHeight = totalH + 'px';

        // Background
        const bgR = this._el('rect');
        bgR.setAttribute('x', 0); bgR.setAttribute('y', 0);
        bgR.setAttribute('width', CPR.SVG_W); bgR.setAttribute('height', totalH);
        bgR.setAttribute('fill', '#070808');
        svg.appendChild(bgR);

        // Floors + connectors
        sorted.forEach((lv, i) => {
            svg.appendChild(this._buildFloor(lv, state, drivingVehicles));
            if (i < sorted.length - 1) svg.appendChild(this._buildConnector(lv, sorted[i + 1]));
        });

        // Entry queue
        if (state.entry_queue?.length) {
            this.renderEntryQueue(svg, state.entry_queue, this.levelOffsets);
        }

        // Animation layer on top
        const animLayer = existingAnimLayer || (() => {
            const g = this._el('g'); g.setAttribute('id', 'cpr-anim-layer'); return g;
        })();
        svg.appendChild(animLayer);
        this.animLayer = animLayer;

        container.innerHTML = '';
        container.appendChild(svg);
        this.svg = svg;

        this._bindEvents(svg);
    }

    // ── Floor group ───────────────────────────────────────────────
    _buildFloor(level, state, drivingVehicles) {
        const baysPerRow = Math.max(...level.bays.map(b => b.position));
        const bayW  = CPR.USABLE_W / baysPerRow;
        const yOff  = this.levelOffsets[level.level_number];
        const isMain = (level.level_number === 1);

        const traffic  = state.traffic || {};
        const zoneOcc  = traffic.zone_occupancy || {};
        const zoneCap  = traffic.zone_capacity  || {};
        const gateLoad = zoneCap.entry_gate > 0
            ? (zoneOcc.entry_gate || 0) / zoneCap.entry_gate : 0;

        const g = this._el('g');
        g.setAttribute('id', `floor-g-${level.level_number}`);
        g.setAttribute('transform', `translate(0,${yOff})`);

        // ── Background panel ──
        const panel = this._el('rect');
        panel.setAttribute('x', CPR.MARGIN_L); panel.setAttribute('y', CPR.LABEL_H);
        panel.setAttribute('width', CPR.USABLE_W);
        panel.setAttribute('height', CPR.FLOOR_H - CPR.LABEL_H);
        panel.setAttribute('fill', '#0c0c10');
        panel.setAttribute('stroke', '#2a1a00'); panel.setAttribute('stroke-width', '1.5');
        g.appendChild(panel);

        // ── Header bar ──
        const hdr = this._el('rect');
        hdr.setAttribute('x', CPR.MARGIN_L); hdr.setAttribute('y', 0);
        hdr.setAttribute('width', CPR.USABLE_W); hdr.setAttribute('height', CPR.LABEL_H);
        hdr.setAttribute('fill', '#120c00');
        hdr.setAttribute('stroke', '#4a2e00'); hdr.setAttribute('stroke-width', '1');
        g.appendChild(hdr);

        g.appendChild(this._text(
            CPR.MARGIN_L + 12, CPR.LABEL_H / 2 + 5,
            level.name.toUpperCase(),
            { fill: '#ff8800', size: 11, weight: 600, font: 'Space Grotesk,sans-serif', letterSpacing: '2' }
        ));

        const occPct = (level.occupancy_rate * 100).toFixed(0);
        g.appendChild(this._text(
            CPR.SVG_W - CPR.MARGIN_R - 8, CPR.LABEL_H / 2 + 5,
            `${level.occupied_count}/${level.capacity}  ${occPct}%`,
            { fill: '#ffcc00', size: 10, font: 'Space Mono,monospace', anchor: 'end' }
        ));

        // ── Aisles ──
        g.appendChild(this._aisle(CPR.AISLE1_Y, CPR.AISLE_H, gateLoad, 1));
        g.appendChild(this._aisle(CPR.AISLE2_Y, CPR.AISLE_H, gateLoad * 0.6, 2));

        // Right corridor connecting both aisles
        const corrX = CPR.MARGIN_L + CPR.USABLE_W - 34;
        const corrRect = this._el('rect');
        corrRect.setAttribute('x', corrX);
        corrRect.setAttribute('y', CPR.AISLE1_Y);
        corrRect.setAttribute('width', 34);
        corrRect.setAttribute('height', CPR.AISLE2_Y + CPR.AISLE_H - CPR.AISLE1_Y);
        corrRect.setAttribute('fill', '#0a160a');
        g.appendChild(corrRect);
        // corridor centre line
        const corrLine = this._el('line');
        corrLine.setAttribute('x1', corrX + 17); corrLine.setAttribute('y1', CPR.AISLE1_Y + 4);
        corrLine.setAttribute('x2', corrX + 17); corrLine.setAttribute('y2', CPR.AISLE2_Y + CPR.AISLE_H - 4);
        corrLine.setAttribute('stroke', 'rgba(255,210,0,0.12)');
        corrLine.setAttribute('stroke-width', '1');
        corrLine.setAttribute('stroke-dasharray', '6,5');
        g.appendChild(corrLine);

        // ── Realistic bay stripe dividers ──
        this._addBayDividers(g, baysPerRow, bayW);

        // ── Pillars ──
        this._addPillars(g, baysPerRow, bayW);

        // ── Lane direction arrows ──
        this._addAisleArrows(g, CPR.AISLE1_Y, CPR.AISLE_H, true);
        this._addAisleArrows(g, CPR.AISLE2_Y, CPR.AISLE_H, false);

        // ── Entry / exit / ramp labels ──
        if (isMain) {
            this._addEntryGate(g);
        }
        if (level.level_number > -1) {
            g.appendChild(this._text(4, CPR.AISLE1_CY + 4,
                '↓RAMP', { fill: '#4488ff', size: 9, font: 'Space Mono,monospace' }));
        }

        // Shop gate marker
        g.appendChild(this._text(CPR.SVG_W / 2, CPR.LABEL_H + 14,
            '⬡ SHOPS',
            { fill: 'rgba(255,136,0,0.4)', size: 9, font: 'Space Mono,monospace', anchor: 'middle' }));

        // ── Bays ──
        const vByBay = {};
        // Parked vehicles (fully arrived)
        Object.values(state.active_vehicles || {}).forEach(v => {
            if (v.assigned_level === level.level_number && v.assigned_bay_id)
                vByBay[v.assigned_bay_id] = { ...v, _transit: false };
        });
        // In-transit vehicles (driving to their bay) — show as ghost dots
        Object.values(state.in_transit_vehicles || {}).forEach(info => {
            const v = info.vehicle;
            if (info.level === level.level_number && info.bay_id && !vByBay[info.bay_id]) {
                // Only show ghost if not currently being animated (animated cars show as moving dots)
                if (!drivingVehicles.has(v.id)) {
                    vByBay[info.bay_id] = { ...v, _transit: true };
                }
            }
        });

        level.bays.forEach(bay => {
            const rowY = CPR.ROW_Y[bay.row];
            if (rowY === undefined) return;
            const x      = CPR.MARGIN_L + (bay.position - 1) * bayW;
            const fd     = CPR.ROW_FACES_DOWN[bay.row];
            const veh    = vByBay[bay.id] || null;
            // Skip the static dot if the vehicle is currently animating (has its own moving dot)
            const skipV  = veh && !veh._transit && drivingVehicles.has(veh.id);
            g.appendChild(this._bay(bay, x, rowY, bayW, fd, skipV ? null : veh));
        });

        // Row labels
        ['A','B','C','D'].forEach(row => {
            const ry = CPR.ROW_Y[row];
            if (ry === undefined) return;
            g.appendChild(this._text(
                CPR.MARGIN_L - 6, ry + CPR.BAY_H / 2 + 4, row,
                { fill: 'rgba(255,136,0,0.45)', size: 10, font: 'Space Grotesk,sans-serif',
                  weight: 700, anchor: 'end' }
            ));
        });

        return g;
    }

    // ── Aisle with congestion tint + dashed centre ────────────────
    _aisle(y, h, congestion = 0, aisleNum = 1) {
        const g = this._el('g');

        const r = this._el('rect');
        r.setAttribute('x', CPR.MARGIN_L); r.setAttribute('y', y);
        r.setAttribute('width', CPR.USABLE_W); r.setAttribute('height', h);
        const alpha = 0.04 + congestion * 0.16;
        r.setAttribute('fill', congestion > 0.5
            ? `rgba(255,100,0,${alpha})`
            : '#0a1708');
        g.appendChild(r);

        // Dashed centre line
        const cy   = y + h / 2;
        const dash = this._el('line');
        dash.setAttribute('x1', CPR.MARGIN_L + 8); dash.setAttribute('y1', cy);
        dash.setAttribute('x2', CPR.MARGIN_L + CPR.USABLE_W - 40); dash.setAttribute('y2', cy);
        dash.setAttribute('stroke', congestion > 0.5 ? 'rgba(255,160,0,0.28)' : 'rgba(255,210,0,0.14)');
        dash.setAttribute('stroke-width', '1');
        dash.setAttribute('stroke-dasharray', '10,8');
        g.appendChild(dash);

        return g;
    }

    // ── Lane direction arrows in aisle ────────────────────────────
    _addAisleArrows(g, aisleY, aisleH, rightward) {
        const cy       = aisleY + aisleH / 2;
        const count    = 5;
        const step     = (CPR.USABLE_W - 60) / count;
        const baseX    = CPR.MARGIN_L + 30;
        const arrowCol = 'rgba(255,220,0,0.18)';

        for (let i = 0; i < count; i++) {
            const ax = baseX + i * step + step / 2;
            const ar = 7, ah = 4;
            let pts;
            if (rightward) {
                pts = `${ax - ar},${cy - ah} ${ax + ar},${cy} ${ax - ar},${cy + ah} ${ax - ar + 4},${cy}`;
            } else {
                pts = `${ax + ar},${cy - ah} ${ax - ar},${cy} ${ax + ar},${cy + ah} ${ax + ar - 4},${cy}`;
            }
            const arrow = this._el('polygon');
            arrow.setAttribute('points', pts);
            arrow.setAttribute('fill', arrowCol);
            arrow.setAttribute('pointer-events', 'none');
            g.appendChild(arrow);
        }
    }

    // ── Thin stripe dividers between bays ─────────────────────────
    _addBayDividers(g, baysPerRow, bayW) {
        const rows = ['A','B','C','D'];
        rows.forEach(row => {
            const ry = CPR.ROW_Y[row];
            for (let i = 1; i < baysPerRow; i++) {
                const dx = CPR.MARGIN_L + i * bayW;
                const line = this._el('line');
                line.setAttribute('x1', dx); line.setAttribute('y1', ry + 1);
                line.setAttribute('x2', dx); line.setAttribute('y2', ry + CPR.BAY_H - 1);
                line.setAttribute('stroke', 'rgba(255,255,255,0.06)');
                line.setAttribute('stroke-width', '0.5');
                line.setAttribute('pointer-events', 'none');
                g.appendChild(line);
            }
        });
    }

    // ── Column/pillar markers at structural grid points ───────────
    _addPillars(g, baysPerRow, bayW) {
        const xs = [0, Math.floor(baysPerRow / 2), baysPerRow].map(
            n => CPR.MARGIN_L + n * bayW
        );
        const ys = [CPR.LABEL_H, CPR.AISLE2_Y + CPR.AISLE_H, CPR.ROW_Y.D + CPR.BAY_H];

        xs.forEach(px => ys.forEach(py => {
            const sq = this._el('rect');
            sq.setAttribute('x', px - 4); sq.setAttribute('y', py - 4);
            sq.setAttribute('width', 8); sq.setAttribute('height', 8);
            sq.setAttribute('fill', '#1e1e2a');
            sq.setAttribute('stroke', '#44446a');
            sq.setAttribute('stroke-width', '1');
            sq.setAttribute('pointer-events', 'none');
            g.appendChild(sq);
        }));
    }

    // ── Realistic entry/exit gate on Level 1 ─────────────────────
    _addEntryGate(g) {
        const gateW = 38;
        const gateX = CPR.MARGIN_L + CPR.USABLE_W - gateW;
        const gateY = CPR.AISLE1_Y + 3;
        const gateH = CPR.AISLE_H - 6;

        // Entry lane (green)
        const entryBox = this._el('rect');
        entryBox.setAttribute('x', gateX); entryBox.setAttribute('y', gateY);
        entryBox.setAttribute('width', gateW / 2 - 1); entryBox.setAttribute('height', gateH);
        entryBox.setAttribute('fill', 'rgba(30,90,30,0.45)');
        entryBox.setAttribute('stroke', '#44ff66'); entryBox.setAttribute('stroke-width', '0.8');
        g.appendChild(entryBox);

        // Exit lane (red)
        const exitBox = this._el('rect');
        exitBox.setAttribute('x', gateX + gateW / 2 + 1); exitBox.setAttribute('y', gateY);
        exitBox.setAttribute('width', gateW / 2 - 1); exitBox.setAttribute('height', gateH);
        exitBox.setAttribute('fill', 'rgba(90,20,20,0.45)');
        exitBox.setAttribute('stroke', '#ff4444'); exitBox.setAttribute('stroke-width', '0.8');
        g.appendChild(exitBox);

        // Barrier stripe (centre divider)
        for (let i = 0; i < 4; i++) {
            const stripe = this._el('rect');
            stripe.setAttribute('x', gateX + gateW / 2 - 1);
            stripe.setAttribute('y', gateY + i * (gateH / 4));
            stripe.setAttribute('width', 2);
            stripe.setAttribute('height', gateH / 4);
            stripe.setAttribute('fill', i % 2 === 0 ? '#ffee00' : '#333300');
            stripe.setAttribute('pointer-events', 'none');
            g.appendChild(stripe);
        }

        // Labels
        g.appendChild(this._text(
            gateX + gateW / 4, gateY + gateH / 2 + 3,
            '◀IN', { fill: '#44ff66', size: 7, font: 'Space Mono,monospace', anchor: 'middle' }
        ));
        g.appendChild(this._text(
            gateX + gateW * 0.75, gateY + gateH / 2 + 3,
            'OUT▶', { fill: '#ff4444', size: 7, font: 'Space Mono,monospace', anchor: 'middle' }
        ));
    }

    // ── Single bay ────────────────────────────────────────────────
    _bay(bay, x, y, bayW, faceDown, vehicle) {
        const g    = this._el('g');
        const isOcc = bay.status === 'occupied';
        const isRes = bay.status === 'reserved';
        const cols  = isOcc ? BAY_COLOURS.occupied :
                      isRes ? BAY_COLOURS.reserved :
                      (BAY_COLOURS[bay.bay_type] || BAY_COLOURS.standard);

        // Bay rectangle
        const r = this._el('rect');
        r.setAttribute('x', x + 0.5); r.setAttribute('y', y);
        r.setAttribute('width', bayW - 1); r.setAttribute('height', CPR.BAY_H);
        r.setAttribute('fill', cols.fill);
        r.setAttribute('stroke', cols.stroke);
        r.setAttribute('stroke-width', '0.8');
        r.setAttribute('class', `bay bay-${bay.status} bay-type-${bay.bay_type}`);
        r.setAttribute('data-bay-id', bay.id);
        r.style.cursor = 'pointer';
        g.appendChild(r);

        // Aisle-facing accent line
        const lineY = faceDown ? y + CPR.BAY_H - 1 : y + 1;
        const accent = this._el('line');
        accent.setAttribute('x1', x + 2); accent.setAttribute('y1', lineY);
        accent.setAttribute('x2', x + bayW - 2); accent.setAttribute('y2', lineY);
        accent.setAttribute('stroke', cols.stroke);
        accent.setAttribute('stroke-width', '2');
        accent.setAttribute('pointer-events', 'none');
        g.appendChild(accent);

        // Reserved dashed border
        if (isRes && bayW > 20) {
            const dr = this._el('rect');
            dr.setAttribute('x', x + 2); dr.setAttribute('y', y + 2);
            dr.setAttribute('width', bayW - 4); dr.setAttribute('height', CPR.BAY_H - 4);
            dr.setAttribute('fill', 'none');
            dr.setAttribute('stroke', 'rgba(255,180,0,0.4)');
            dr.setAttribute('stroke-width', '1');
            dr.setAttribute('stroke-dasharray', '4,3');
            dr.setAttribute('pointer-events', 'none');
            g.appendChild(dr);
        }

        // Bay number
        if (bayW > 22) {
            const numY = faceDown ? y + CPR.BAY_H - 6 : y + 11;
            g.appendChild(this._text(x + bayW / 2, numY, String(bay.position), {
                fill: 'rgba(255,136,0,0.3)', size: 7,
                font: 'Space Mono,monospace', anchor: 'middle', ptrEvt: 'none'
            }));
        }

        // Special bay icon
        if (!isOcc && !isRes && bay.bay_type !== 'standard' && bayW > 26) {
            const icons = { blue_badge: '♿', parent_child: '👶', ev: '⚡' };
            const ic = icons[bay.bay_type];
            if (ic) {
                const icY = faceDown ? y + 13 : y + CPR.BAY_H - 6;
                g.appendChild(this._text(x + bayW / 2, icY, ic, {
                    fill: cols.stroke, size: 9, anchor: 'middle', ptrEvt: 'none'
                }));
            }
        }

        // ── Parked / in-transit vehicle dot ──
        if (vehicle) {
            const cx  = x + bayW / 2;
            const cy  = faceDown
                ? y + CPR.BAY_H * 0.36
                : y + CPR.BAY_H * 0.64;
            const r0  = Math.min(bayW * 0.30, CPR.BAY_H * 0.30);
            const isTransit = vehicle._transit === true;
            // Stagger breathe timing by bay id charcode so they don't all pulse together
            const phase = (bay.id.charCodeAt(bay.id.length - 1) % 8) * 0.18;
            const dur   = (isTransit ? 0.7 + phase * 0.3 : 1.6 + phase).toFixed(2) + 's';

            // Outer glow (dimmer for in-transit)
            const glow = this._el('circle');
            glow.setAttribute('cx', cx); glow.setAttribute('cy', cy);
            glow.setAttribute('r', r0 * 1.9);
            glow.setAttribute('fill', vehicle.color);
            glow.setAttribute('opacity', isTransit ? '0.06' : '0.10');
            glow.setAttribute('pointer-events', 'none');
            g.appendChild(glow);

            if (!isTransit) {
                // Animated glow opacity (parked only)
                const animGO = this._el('animate');
                animGO.setAttribute('attributeName', 'opacity');
                animGO.setAttribute('values', '0.10;0.22;0.10');
                animGO.setAttribute('dur', dur);
                animGO.setAttribute('repeatCount', 'indefinite');
                glow.appendChild(animGO);
            }

            // Main dot
            const dot = this._el('circle');
            dot.setAttribute('cx', cx); dot.setAttribute('cy', cy);
            dot.setAttribute('r', isTransit ? (r0 * 0.7).toFixed(1) : r0);
            dot.setAttribute('fill', vehicle.color);
            dot.setAttribute('opacity', isTransit ? '0.45' : '1');
            if (!isTransit) {
                dot.setAttribute('class', 'sv-vehicle');
                dot.setAttribute('data-vehicle-id', vehicle.id);
                dot.style.cursor = 'pointer';
            } else {
                dot.setAttribute('pointer-events', 'none');
            }

            // Breathe radius (parked = smooth; transit = rapid pulse)
            const animR = this._el('animate');
            animR.setAttribute('attributeName', 'r');
            if (isTransit) {
                const r1 = (r0 * 0.7).toFixed(1), r2 = (r0 * 0.85).toFixed(1);
                animR.setAttribute('values', `${r1};${r2};${r1}`);
            } else {
                animR.setAttribute('values', `${r0.toFixed(1)};${(r0 * 1.22).toFixed(1)};${r0.toFixed(1)}`);
            }
            animR.setAttribute('dur', dur);
            animR.setAttribute('repeatCount', 'indefinite');
            dot.appendChild(animR);

            // Breathe opacity
            const animO = this._el('animate');
            animO.setAttribute('attributeName', 'opacity');
            animO.setAttribute('values', isTransit ? '0.45;0.20;0.45' : '1;0.72;1');
            animO.setAttribute('dur', dur);
            animO.setAttribute('repeatCount', 'indefinite');
            dot.appendChild(animO);

            g.appendChild(dot);

            if (!isTransit) {
                // Centre specular highlight (parked only)
                const hl = this._el('circle');
                hl.setAttribute('cx', cx); hl.setAttribute('cy', cy - r0 * 0.2);
                hl.setAttribute('r', (r0 * 0.28).toFixed(1));
                hl.setAttribute('fill', 'rgba(255,255,255,0.50)');
                hl.setAttribute('pointer-events', 'none');
                g.appendChild(hl);
            }
        }

        return g;
    }

    // ── Entry queue circles ───────────────────────────────────────
    renderEntryQueue(svg, entryQueue, levelOffsets) {
        if (!entryQueue || !entryQueue.length) return;
        const level1Y = levelOffsets[1] || 0;
        const queueX  = CPR.SVG_W - CPR.MARGIN_R + 8;
        const baseY   = level1Y + CPR.AISLE1_CY;

        svg.appendChild(this._text(
            queueX + 10, baseY - 24,
            `Q(${entryQueue.length})`,
            { fill: '#ffaa00', size: 7.5, font: 'Space Mono,monospace', anchor: 'middle' }
        ));

        const visible = entryQueue.slice(0, 9);
        visible.forEach((v, i) => {
            const cy = baseY - 8 - i * 13;
            const c  = this._el('circle');
            c.setAttribute('cx', queueX + 10); c.setAttribute('cy', cy);
            c.setAttribute('r', '5');
            c.setAttribute('fill', v.color || '#ff8800');
            c.setAttribute('opacity', '0.9');
            svg.appendChild(c);

            // tiny highlight
            const h = this._el('circle');
            h.setAttribute('cx', queueX + 9); h.setAttribute('cy', cy - 2);
            h.setAttribute('r', '1.5');
            h.setAttribute('fill', 'rgba(255,255,255,0.45)');
            h.setAttribute('pointer-events', 'none');
            svg.appendChild(h);
        });

        if (entryQueue.length > 9) {
            svg.appendChild(this._text(
                queueX + 10, baseY - 8 - 9 * 13 - 5,
                `+${entryQueue.length - 9}`,
                { fill: '#ffaa00', size: 7, font: 'Space Mono,monospace', anchor: 'middle' }
            ));
        }
    }

    // ── Ramp connector ────────────────────────────────────────────
    _buildConnector(upperLevel, lowerLevel) {
        const g = this._el('g');
        const y = this.levelOffsets[upperLevel.level_number] + CPR.FLOOR_H;

        const bg = this._el('rect');
        bg.setAttribute('x', 0); bg.setAttribute('y', y);
        bg.setAttribute('width', CPR.SVG_W); bg.setAttribute('height', CPR.CONNECTOR_H);
        bg.setAttribute('fill', '#040510');
        g.appendChild(bg);

        // Ramp shaft
        const shaftX = CPR.MARGIN_L - 24;
        const shaftW = 68;
        const shaft  = this._el('rect');
        shaft.setAttribute('x', shaftX); shaft.setAttribute('y', y);
        shaft.setAttribute('width', shaftW); shaft.setAttribute('height', CPR.CONNECTOR_H);
        shaft.setAttribute('fill', '#06081c');
        shaft.setAttribute('stroke', '#1a2a52'); shaft.setAttribute('stroke-width', '1');
        g.appendChild(shaft);

        // Diagonal hatch lines inside shaft
        for (let hy = y - 5; hy < y + CPR.CONNECTOR_H + 8; hy += 8) {
            const h = this._el('line');
            h.setAttribute('x1', shaftX); h.setAttribute('y1', hy);
            h.setAttribute('x2', shaftX + shaftW); h.setAttribute('y2', hy + 8);
            h.setAttribute('stroke', 'rgba(68,136,255,0.10)');
            h.setAttribute('stroke-width', '1');
            g.appendChild(h);
        }

        // Speed lines suggesting descent
        [0.25, 0.5, 0.75].forEach(frac => {
            const lx = shaftX + 8 + frac * (shaftW - 16);
            const sl = this._el('line');
            sl.setAttribute('x1', lx); sl.setAttribute('y1', y + 8);
            sl.setAttribute('x2', lx - 4); sl.setAttribute('y2', y + CPR.CONNECTOR_H - 8);
            sl.setAttribute('stroke', 'rgba(68,136,255,0.18)');
            sl.setAttribute('stroke-width', '1.5');
            g.appendChild(sl);
        });

        g.appendChild(this._text(
            shaftX + shaftW / 2, y + CPR.CONNECTOR_H / 2 + 4,
            '↕ RAMP',
            { fill: '#4488ff', size: 9.5, font: 'Space Mono,monospace', anchor: 'middle' }
        ));

        return g;
    }

    // ── Vehicle path computation ──────────────────────────────────
    computeEntryPath(vehicle, targetLevelNum, bayId, levels) {
        const tl = levels.find(l => l.level_number === targetLevelNum);
        if (!tl) return null;
        const bay = tl.bays.find(b => b.id === bayId);
        if (!bay) return null;
        return this._buildPath(bay, targetLevelNum, levels, 'entry');
    }

    computeExitPath(vehicle, targetLevelNum, bayId, levels) {
        const tl = levels.find(l => l.level_number === targetLevelNum);
        if (!tl) return null;
        const bay = tl.bays.find(b => b.id === bayId);
        if (!bay) return null;
        return this._buildPath(bay, targetLevelNum, levels, 'exit');
    }

    _buildPath(bay, targetLevelNum, levels, direction) {
        const baysPerRow = Math.max(...levels.find(l => l.level_number === targetLevelNum).bays.map(b => b.position));
        const bayW    = CPR.USABLE_W / baysPerRow;
        const bayX    = CPR.MARGIN_L + (bay.position - 1) * bayW + bayW / 2;
        const row     = bay.row;
        const fd      = CPR.ROW_FACES_DOWN[row];
        const aisleN  = CPR.ROW_AISLE[row] || 1;

        const tFloorY  = this.levelOffsets[targetLevelNum] || 0;
        const e1FloorY = this.levelOffsets[1]              || 0;

        const tA1Y   = tFloorY  + CPR.AISLE1_CY;
        const tA2Y   = tFloorY  + CPR.AISLE2_CY;
        const e1A1Y  = e1FloorY + CPR.AISLE1_CY;
        const aisleY = aisleN === 1 ? tA1Y : tA2Y;

        const baySlotY = tFloorY + CPR.ROW_Y[row] + (fd ? CPR.BAY_H * 0.28 : CPR.BAY_H * 0.72);

        const rightEntryX = CPR.SVG_W - CPR.MARGIN_R;
        const offscreenX  = CPR.SVG_W + 30;
        const rampX       = CPR.MARGIN_L;

        // Ramp chain ends at (rampX, tA1Y) — skip first aisleN===1 point if ramp already lands there
        const rampPts  = targetLevelNum < 1 ? this._rampChain(targetLevelNum, levels) : [];
        const aislePts = aisleN === 2
            ? [
                { x: rampX,       y: tA1Y  },
                { x: rightEntryX, y: tA1Y  },
                { x: rightEntryX, y: tA2Y  },
                { x: bayX,        y: tA2Y  },
              ]
            : [
                // If ramp already put us at rampX/tA1Y, skip the duplicate
                ...(rampPts.length === 0 ? [{ x: rampX, y: tA1Y }] : []),
                { x: bayX, y: tA1Y },
              ];

        const entryPath = [
            { x: rightEntryX, y: e1A1Y },
            ...rampPts,
            ...aislePts,
            { x: bayX, y: baySlotY },
        ];

        if (direction === 'entry') return entryPath;
        const exit = [...entryPath].reverse();
        return exit;
    }

    _rampChain(targetLevelNum, levels) {
        const pts = [];
        let cur = 1;
        while (cur > targetLevelNum) {
            const curY  = (this.levelOffsets[cur]   || 0) + CPR.AISLE1_CY;
            const nextY = (this.levelOffsets[cur-1] || 0) + CPR.AISLE1_CY;
            pts.push({ x: CPR.MARGIN_L, y: curY  });
            pts.push({ x: CPR.MARGIN_L, y: nextY });
            cur--;
        }
        return pts;
    }

    // ── SVG helpers ───────────────────────────────────────────────
    _el(tag) { return document.createElementNS(this.NS, tag); }

    _text(x, y, content, opts = {}) {
        const t = this._el('text');
        t.setAttribute('x', x); t.setAttribute('y', y);
        t.setAttribute('fill', opts.fill || '#ff8800');
        t.setAttribute('font-size', opts.size || 10);
        t.setAttribute('font-family', opts.font || 'Space Mono,monospace');
        if (opts.weight)        t.setAttribute('font-weight', opts.weight);
        if (opts.anchor)        t.setAttribute('text-anchor', opts.anchor);
        if (opts.letterSpacing) t.setAttribute('letter-spacing', opts.letterSpacing);
        if (opts.ptrEvt)        t.setAttribute('pointer-events', opts.ptrEvt);
        t.textContent = content;
        return t;
    }

    _bindEvents(svg) {
        svg.querySelectorAll('.bay').forEach(el => {
            el.addEventListener('click', e => {
                e.stopPropagation();
                if (this._callbacks.bay) {
                    const bayId  = el.getAttribute('data-bay-id');
                    const lvlAttr = el.closest('g[id^="floor-g-"]')?.id.replace('floor-g-', '');
                    this._callbacks.bay(e, bayId, parseInt(lvlAttr ?? '0', 10));
                }
            });
        });
        svg.querySelectorAll('.sv-vehicle').forEach(el => {
            el.addEventListener('click', e => {
                e.stopPropagation();
                if (this._callbacks.vehicle)
                    this._callbacks.vehicle(e, el.getAttribute('data-vehicle-id'));
            });
        });
    }
}


// ── VehicleAnimator ──────────────────────────────────────────────────
class VehicleAnimator {
    constructor(getAnimLayer) {
        this._getLayer  = getAnimLayer;
        this.NS         = 'http://www.w3.org/2000/svg';
        this._active    = new Map();
        this._rafId     = null;
        this._tick      = this._tick.bind(this);
        this._onNearMiss    = null;
        this._nearMissCount = 0;
        this._speedScale    = 1.0;   // multiplied onto every vehicle's per-frame t increment
    }

    onNearMiss(fn) { this._onNearMiss = fn; }

    /** Set a global speed multiplier (1 = normal, >1 = faster animations). */
    setSpeedScale(scale) { this._speedScale = Math.max(0.1, scale); }

    // ── Start an animation ────────────────────────────────────────
    start(vehicle, path, onDone) {
        if (!path || path.length < 2) { if (onDone) onDone(); return; }
        this.remove(vehicle.id);

        const layer = this._getLayer();
        if (!layer) return;

        // Glow ring behind dot
        const glow = this._makeEl('circle', {
            r: 13, fill: vehicle.color,
            opacity: 0.18, 'pointer-events': 'none',
        });
        // Main dot
        const dot = this._makeEl('circle', {
            r: 7, fill: vehicle.color,
            class: 'av-body',
        });
        // Specular highlight
        const hl = this._makeEl('circle', {
            r: 2.2, fill: 'rgba(255,255,255,0.55)',
            'pointer-events': 'none',
        });

        layer.appendChild(glow);
        layer.appendChild(dot);
        layer.appendChild(hl);

        const distOf = i => {
            const a = path[i], b = path[i + 1] || a;
            return Math.hypot(b.x - a.x, b.y - a.y);
        };
        const scale = this._speedScale;
        // Slower base speed for smoother motion; scaled by _speedScale during training
        const speedFor = dist => Math.max(0.006, Math.min(0.9, (48 / Math.max(dist, 1)) * scale));

        const state = {
            dot, hl, glow, path,
            segIdx: 0, t: 0,
            speed: speedFor(distOf(0)),
            done: false, onDone,
            _origGlowColor: vehicle.color,
            _braking: false,
            _nearMissReported: false,
        };

        this._active.set(vehicle.id, state);
        this._place(state, path[0].x, path[0].y);

        if (!this._rafId) this._rafId = requestAnimationFrame(this._tick);
    }

    // ── Animation tick ────────────────────────────────────────────
    _tick() {
        for (const [vid, a] of this._active.entries()) {
            if (a.done) {
                [a.dot, a.hl, a.glow].forEach(e => e.parentNode?.removeChild(e));
                this._active.delete(vid);
                if (a.onDone) a.onDone();
                continue;
            }

            // Cubic-eased progress
            a.t += a.speed;
            if (a.t >= 1) {
                a.t = 0;
                a.segIdx++;
                if (a.segIdx >= a.path.length - 1) { a.done = true; continue; }
                const dist = Math.hypot(
                    a.path[a.segIdx + 1].x - a.path[a.segIdx].x,
                    a.path[a.segIdx + 1].y - a.path[a.segIdx].y
                );
                a.speed = Math.max(0.006, Math.min(0.9, (48 / Math.max(dist, 1)) * this._speedScale));
            }

            const from = a.path[a.segIdx];
            const to   = a.path[Math.min(a.segIdx + 1, a.path.length - 1)];
            const et   = this._ease(a.t);
            const x    = from.x + (to.x - from.x) * et;
            const y    = from.y + (to.y - from.y) * et;
            this._place(a, x, y);
        }

        // ── Proximity / near-miss ─────────────────────────────────
        const SAFE_DIST  = 24;
        const BRAKE_DIST = 14;
        const entries = [...this._active.entries()].filter(([, a]) => !a.done);

        for (let i = 0; i < entries.length; i++) {
            for (let j = i + 1; j < entries.length; j++) {
                const [, ai] = entries[i];
                const [, aj] = entries[j];
                if (ai.cx == null || aj.cx == null) continue;

                const dx   = ai.cx - aj.cx;
                const dy   = ai.cy - aj.cy;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < SAFE_DIST) {
                    const trailing = (ai.segIdx < aj.segIdx || (ai.segIdx === aj.segIdx && ai.t < aj.t)) ? ai : aj;
                    const maxSpeed = Math.max(0.002, (dist / SAFE_DIST) * 0.035);
                    trailing.speed = Math.min(trailing.speed, maxSpeed);

                    if (!trailing._braking) {
                        trailing._braking = true;
                        trailing.glow.setAttribute('fill', '#ff2200');
                        trailing.glow.setAttribute('opacity', '0.45');
                    }
                    if (dist < BRAKE_DIST && !trailing._nearMissReported) {
                        trailing._nearMissReported = true;
                        if (this._onNearMiss) this._onNearMiss();
                    }
                } else {
                    for (const [, a] of [entries[i], entries[j]]) {
                        if (a._braking) {
                            a._braking = false;
                            a.glow.setAttribute('fill', a._origGlowColor);
                            a.glow.setAttribute('opacity', '0.18');
                        }
                    }
                }
            }
        }

        if (this._active.size > 0) {
            this._rafId = requestAnimationFrame(this._tick);
        } else {
            this._rafId = null;
        }
    }

    // ── Position dot elements ─────────────────────────────────────
    _place(a, x, y) {
        a.cx = x; a.cy = y;
        a.dot.setAttribute('cx',  x); a.dot.setAttribute('cy',  y);
        a.glow.setAttribute('cx', x); a.glow.setAttribute('cy', y);
        a.hl.setAttribute('cx',   x); a.hl.setAttribute('cy',   y - 2);
    }

    // ── Helpers ───────────────────────────────────────────────────
    _makeEl(tag, attrs) {
        const el = document.createElementNS(this.NS, tag);
        Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
        return el;
    }

    _ease(t) {
        // Smooth cubic ease-in-out
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    remove(vehicleId) {
        const a = this._active.get(vehicleId);
        if (a) {
            [a.dot, a.hl, a.glow].forEach(e => e.parentNode?.removeChild(e));
            this._active.delete(vehicleId);
        }
    }

    has(vehicleId) { return this._active.has(vehicleId); }

    clear() {
        for (const a of this._active.values()) {
            [a.dot, a.hl, a.glow].forEach(e => e.parentNode?.removeChild(e));
        }
        this._active.clear();
    }
}
