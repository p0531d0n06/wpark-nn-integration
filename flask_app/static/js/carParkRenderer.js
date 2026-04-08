/**
 * WPARK – SVG Parking Plan Renderer
 * Renders all floors as one connected top-down plan (inspired by Parking Plan -3)
 * Layout: two row pairs per floor, each pair sharing an aisle
 */

// ── Layout constants ───────────────────────────────────────────────
const CPR = {
    SVG_W:       860,
    MARGIN_L:    58,    // left edge – ramp labels live here
    MARGIN_R:    42,    // right edge – entry / exit labels live here
    BAY_H:       40,    // bay rectangle height  (top-down depth)
    AISLE_H:     44,    // driving aisle height
    LABEL_H:     24,    // floor-name header strip
    CONNECTOR_H: 54,    // ramp section between floors
    PAD_BOTTOM:  8,
};

CPR.USABLE_W = CPR.SVG_W - CPR.MARGIN_L - CPR.MARGIN_R;   // 760 px

// Row top Y positions relative to the floor group origin
//   LABEL | A | AISLE1 | B | C | AISLE2 | D | PADDING
CPR.ROW_Y = {
    A: CPR.LABEL_H,
    B: CPR.LABEL_H + CPR.BAY_H + CPR.AISLE_H,
    C: CPR.LABEL_H + CPR.BAY_H + CPR.AISLE_H + CPR.BAY_H,
    D: CPR.LABEL_H + CPR.BAY_H + CPR.AISLE_H + CPR.BAY_H + CPR.BAY_H + CPR.AISLE_H,
};

CPR.FLOOR_H   = CPR.ROW_Y.D + CPR.BAY_H + CPR.PAD_BOTTOM; // 280

CPR.AISLE1_Y  = CPR.ROW_Y.A + CPR.BAY_H;                   // 64
CPR.AISLE1_CY = CPR.AISLE1_Y + CPR.AISLE_H / 2;            // 86
CPR.AISLE2_Y  = CPR.ROW_Y.C + CPR.BAY_H;                   // 188
CPR.AISLE2_CY = CPR.AISLE2_Y + CPR.AISLE_H / 2;            // 210

CPR.ROW_FACES_DOWN = { A: true,  B: false, C: true,  D: false };
CPR.ROW_AISLE      = { A: 1,     B: 1,     C: 2,     D: 2    };

// Bay type colours (fill / stroke)
const BAY_COLOURS = {
    standard:     { fill: '#0c1a0c', stroke: '#2a4a1a' },
    blue_badge:   { fill: '#05101f', stroke: '#1a4488' },
    parent_child: { fill: '#0e0718', stroke: '#662299' },
    ev:           { fill: '#051508', stroke: '#1a6622' },
    occupied:     { fill: '#1a0800', stroke: '#884422' },
    reserved:     { fill: '#12110a', stroke: '#665500' },
};

// ── ParkingPlanRenderer ─────────────────────────────────────────────
class ParkingPlanRenderer {
    constructor() {
        this.NS = 'http://www.w3.org/2000/svg';
        this.levelOffsets = {};   // level_number → SVG Y offset
        this.svg = null;
        this.animLayer = null;
        this._callbacks = { bay: null, vehicle: null };
    }

    onBayClick(fn)     { this._callbacks.bay     = fn; }
    onVehicleClick(fn) { this._callbacks.vehicle = fn; }

    // ── Main render call ─────────────────────────────────────
    render(container, levels, state, drivingVehicles = new Set()) {
        // Sort floors: Level 1 at top → Level -1 at bottom
        const sorted = [...levels].sort((a, b) => b.level_number - a.level_number);

        // Compute Y offsets
        this.levelOffsets = {};
        let yOff = 0;
        sorted.forEach((lv, i) => {
            this.levelOffsets[lv.level_number] = yOff;
            yOff += CPR.FLOOR_H + (i < sorted.length - 1 ? CPR.CONNECTOR_H : 0);
        });
        const totalH = yOff;

        // Preserve animation layer across re-renders
        const existingAnimLayer = document.getElementById('cpr-anim-layer');

        // Build new SVG
        const svg = this._el('svg');
        svg.setAttribute('id', 'carpark-svg');
        svg.setAttribute('viewBox', `0 0 ${CPR.SVG_W} ${totalH}`);
        svg.setAttribute('width', '100%');
        svg.style.display = 'block';
        svg.style.minHeight = totalH + 'px';

        // Background
        const bgR = this._el('rect');
        bgR.setAttribute('x', 0); bgR.setAttribute('y', 0);
        bgR.setAttribute('width', CPR.SVG_W); bgR.setAttribute('height', totalH);
        bgR.setAttribute('fill', '#080808');
        svg.appendChild(bgR);

        // Floor groups
        sorted.forEach((lv, i) => {
            svg.appendChild(this._buildFloor(lv, state, drivingVehicles));
            if (i < sorted.length - 1) svg.appendChild(this._buildConnector(lv, sorted[i + 1]));
        });

        // Entry queue visualization
        if (state.entry_queue?.length) {
            this.renderEntryQueue(svg, state.entry_queue, this.levelOffsets);
        }

        // Re-attach or create animation layer on top
        const animLayer = existingAnimLayer || (() => {
            const g = this._el('g'); g.setAttribute('id', 'cpr-anim-layer'); return g;
        })();
        svg.appendChild(animLayer);
        this.animLayer = animLayer;

        container.innerHTML = '';
        container.appendChild(svg);
        this.svg = svg;

        // Bind click events
        this._bindEvents(svg);
    }

    // ── Floor group ───────────────────────────────────────────
    _buildFloor(level, state, drivingVehicles) {
        const baysPerRow = Math.max(...level.bays.map(b => b.position));
        const bayW = CPR.USABLE_W / baysPerRow;
        const yOff = this.levelOffsets[level.level_number];
        const isMain = (level.level_number === 1);

        const traffic = state.traffic || {};
        const zoneOcc = traffic.zone_occupancy || {};
        const zoneCap = traffic.zone_capacity  || {};
        // Derive congestion from entry_gate for Level 1, ramps for others
        const gateLoad = zoneCap.entry_gate > 0
            ? (zoneOcc.entry_gate || 0) / zoneCap.entry_gate : 0;

        const g = this._el('g');
        g.setAttribute('id', `floor-g-${level.level_number}`);
        g.setAttribute('transform', `translate(0,${yOff})`);

        // --- Background panel ---
        const panel = this._el('rect');
        panel.setAttribute('x', CPR.MARGIN_L); panel.setAttribute('y', CPR.LABEL_H);
        panel.setAttribute('width', CPR.USABLE_W);
        panel.setAttribute('height', CPR.FLOOR_H - CPR.LABEL_H);
        panel.setAttribute('fill', '#0e0e12');
        panel.setAttribute('stroke', '#2a1a00'); panel.setAttribute('stroke-width', '1.5');
        g.appendChild(panel);

        // --- Header bar ---
        const hdr = this._el('rect');
        hdr.setAttribute('x', CPR.MARGIN_L); hdr.setAttribute('y', 0);
        hdr.setAttribute('width', CPR.USABLE_W); hdr.setAttribute('height', CPR.LABEL_H);
        hdr.setAttribute('fill', '#150d00');
        hdr.setAttribute('stroke', '#4a2e00'); hdr.setAttribute('stroke-width', '1');
        g.appendChild(hdr);

        g.appendChild(this._text(
            CPR.MARGIN_L + 10, CPR.LABEL_H / 2 + 5,
            level.name.toUpperCase(),
            { fill: '#ff8800', size: 11, weight: 600, font: 'Space Grotesk,sans-serif', letterSpacing: '2' }
        ));

        const occPct = (level.occupancy_rate * 100).toFixed(0);
        g.appendChild(this._text(
            CPR.SVG_W - CPR.MARGIN_R - 6, CPR.LABEL_H / 2 + 5,
            `${level.occupied_count}/${level.capacity}  ${occPct}%`,
            { fill: '#ffcc00', size: 10, font: 'Space Mono,monospace', anchor: 'end' }
        ));

        // --- Aisles ---
        g.appendChild(this._aisle(CPR.AISLE1_Y, CPR.AISLE_H, gateLoad));
        g.appendChild(this._aisle(CPR.AISLE2_Y, CPR.AISLE_H, gateLoad * 0.6));

        // Right corridor connecting aisle 1 and aisle 2
        const corrX = CPR.MARGIN_L + CPR.USABLE_W - 32;
        const corrRect = this._el('rect');
        corrRect.setAttribute('x', corrX);
        corrRect.setAttribute('y', CPR.AISLE1_Y);
        corrRect.setAttribute('width', 32);
        corrRect.setAttribute('height', CPR.AISLE2_Y + CPR.AISLE_H - CPR.AISLE1_Y);
        corrRect.setAttribute('fill', '#0a150a');
        g.appendChild(corrRect);

        // --- Entry / exit / ramp labels ---
        if (isMain) {
            g.appendChild(this._text(CPR.SVG_W - 4, CPR.AISLE1_CY - 4,
                '◀ ENTRY', { fill: '#44ff66', size: 9, font: 'Space Mono,monospace', anchor: 'end' }));
            g.appendChild(this._text(CPR.SVG_W - 4, CPR.AISLE1_CY + 9,
                'EXIT ▶',  { fill: '#ff4444', size: 9, font: 'Space Mono,monospace', anchor: 'end' }));
        }
        if (level.level_number > -1) {
            g.appendChild(this._text(2, CPR.AISLE1_CY + 4,
                '↓RAMP', { fill: '#4488ff', size: 9, font: 'Space Mono,monospace' }));
        }

        // Shop gate marker (top centre)
        g.appendChild(this._text(CPR.SVG_W / 2, CPR.LABEL_H + 12,
            '⬡ SHOPS',
            { fill: 'rgba(255,136,0,0.5)', size: 9, font: 'Space Mono,monospace', anchor: 'middle' }));

        // --- Bays ---
        // Build quick vehicle lookup
        const vByBay = {};
        Object.values(state.active_vehicles || {}).forEach(v => {
            if (v.assigned_level === level.level_number && v.assigned_bay_id)
                vByBay[v.assigned_bay_id] = v;
        });

        level.bays.forEach(bay => {
            const rowY = CPR.ROW_Y[bay.row];
            if (rowY === undefined) return;
            const x   = CPR.MARGIN_L + (bay.position - 1) * bayW;
            const fd  = CPR.ROW_FACES_DOWN[bay.row];
            const veh = vByBay[bay.id] || null;
            const skipVeh = veh && drivingVehicles.has(veh.id);
            g.appendChild(this._bay(bay, x, rowY, bayW, fd, skipVeh ? null : veh));
        });

        // Row labels on left margin
        ['A','B','C','D'].forEach(row => {
            const ry = CPR.ROW_Y[row];
            if (ry === undefined) return;
            g.appendChild(this._text(
                CPR.MARGIN_L - 4, ry + CPR.BAY_H / 2 + 4,
                row,
                { fill: 'rgba(255,136,0,0.45)', size: 10, font: 'Space Grotesk,sans-serif',
                  weight: 700, anchor: 'end' }
            ));
        });

        return g;
    }

    // ── Aisle rectangle + centre dashed line ─────────────────
    _aisle(y, h, congestion = 0) {
        const g = this._el('g');
        const r = this._el('rect');
        r.setAttribute('x', CPR.MARGIN_L); r.setAttribute('y', y);
        r.setAttribute('width', CPR.USABLE_W); r.setAttribute('height', h);
        // Tint from green-dark to amber-dark based on congestion
        const alpha = 0.05 + congestion * 0.18;
        r.setAttribute('fill', congestion > 0.5
            ? `rgba(255,100,0,${alpha})`
            : '#0b180a');
        g.appendChild(r);
        // Centre dashed line
        const cy = y + h / 2;
        const dash = this._el('line');
        dash.setAttribute('x1', CPR.MARGIN_L + 6); dash.setAttribute('y1', cy);
        dash.setAttribute('x2', CPR.MARGIN_L + CPR.USABLE_W - 6); dash.setAttribute('y2', cy);
        dash.setAttribute('stroke', congestion > 0.5 ? 'rgba(255,160,0,0.3)' : 'rgba(255,210,0,0.18)');
        dash.setAttribute('stroke-width', '1');
        dash.setAttribute('stroke-dasharray', '10,8');
        g.appendChild(dash);
        return g;
    }

    // ── Single bay rectangle ──────────────────────────────────
    _bay(bay, x, y, bayW, faceDown, vehicle) {
        const g = this._el('g');
        const isOcc = bay.status === 'occupied';
        const isRes = bay.status === 'reserved';
        const cols = isOcc ? BAY_COLOURS.occupied :
                     isRes ? BAY_COLOURS.reserved :
                     (BAY_COLOURS[bay.bay_type] || BAY_COLOURS.standard);

        // Bay rect
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

        // Aisle-facing edge accent line
        const lineY = faceDown ? y + CPR.BAY_H - 1 : y + 1;
        const accent = this._el('line');
        accent.setAttribute('x1', x + 1.5); accent.setAttribute('y1', lineY);
        accent.setAttribute('x2', x + bayW - 1.5); accent.setAttribute('y2', lineY);
        accent.setAttribute('stroke', cols.stroke);
        accent.setAttribute('stroke-width', '2.5');
        accent.setAttribute('pointer-events', 'none');
        g.appendChild(accent);

        // Reserved bay dashed border indicator
        if (isRes && bayW > 20) {
            const dashedR = this._el('rect');
            dashedR.setAttribute('x', x + 2); dashedR.setAttribute('y', y + 2);
            dashedR.setAttribute('width', bayW - 4); dashedR.setAttribute('height', CPR.BAY_H - 4);
            dashedR.setAttribute('fill', 'none');
            dashedR.setAttribute('stroke', 'rgba(255,180,0,0.5)');
            dashedR.setAttribute('stroke-width', '1');
            dashedR.setAttribute('stroke-dasharray', '4,3');
            dashedR.setAttribute('pointer-events', 'none');
            g.appendChild(dashedR);
        }

        // Bay number
        if (bayW > 22) {
            const numY = faceDown ? y + CPR.BAY_H - 7 : y + 11;
            const lbl = this._text(x + bayW / 2, numY, String(bay.position), {
                fill: 'rgba(255,136,0,0.38)', size: 7,
                font: 'Space Mono,monospace', anchor: 'middle',
                ptrEvt: 'none'
            });
            g.appendChild(lbl);
        }

        // Type icon for special bays
        if (!isOcc && !isRes && bay.bay_type !== 'standard' && bayW > 28) {
            const icons = { blue_badge: '♿', parent_child: '👶', ev: '⚡' };
            const ic = icons[bay.bay_type];
            if (ic) {
                const icY = faceDown ? y + 12 : y + CPR.BAY_H - 7;
                const icT = this._text(x + bayW / 2, icY, ic, {
                    fill: cols.stroke, size: 8, anchor: 'middle', ptrEvt: 'none'
                });
                g.appendChild(icT);
            }
        }

        // Parked vehicle
        if (vehicle) {
            const vw = bayW * 0.7;
            const vh = CPR.BAY_H * 0.55;
            const vx = x + (bayW - vw) / 2;
            const vy = faceDown ? y + CPR.BAY_H * 0.08 : y + CPR.BAY_H * 0.37;

            const vr = this._el('rect');
            vr.setAttribute('x', vx); vr.setAttribute('y', vy);
            vr.setAttribute('width', vw); vr.setAttribute('height', vh);
            vr.setAttribute('fill', vehicle.color);
            vr.setAttribute('rx', '2');
            vr.setAttribute('class', 'sv-vehicle');
            vr.setAttribute('data-vehicle-id', vehicle.id);
            vr.style.cursor = 'pointer';
            g.appendChild(vr);

            // Windshield
            const ws = this._el('rect');
            ws.setAttribute('x', vx + vw * 0.18);
            ws.setAttribute('y', faceDown ? vy + vh * 0.08 : vy + vh * 0.55);
            ws.setAttribute('width', vw * 0.64); ws.setAttribute('height', vh * 0.22);
            ws.setAttribute('fill', 'rgba(255,255,255,0.28)'); ws.setAttribute('rx', '1');
            ws.setAttribute('pointer-events', 'none');
            g.appendChild(ws);
        }

        return g;
    }

    // ── Entry queue visualization ─────────────────────────────
    renderEntryQueue(svg, entryQueue, levelOffsets) {
        if (!entryQueue || !entryQueue.length) return;
        const level1Y = levelOffsets[1] || 0;
        const queueX  = CPR.SVG_W - CPR.MARGIN_R + 4;
        const queueBaseY = level1Y + CPR.AISLE1_CY;

        // Queue label
        const lbl = this._text(queueX + 2, queueBaseY - 18,
            `QUEUE (${entryQueue.length})`,
            { fill: '#ffaa00', size: 7, font: 'Space Mono,monospace' });
        svg.appendChild(lbl);

        // Draw up to 8 queued cars stacked vertically above aisle
        const visible = entryQueue.slice(0, 8);
        visible.forEach((v, i) => {
            const cy = queueBaseY - 8 - i * 14;
            const cr = this._el('rect');
            cr.setAttribute('x', queueX); cr.setAttribute('y', cy - 4);
            cr.setAttribute('width', 20); cr.setAttribute('height', 8);
            cr.setAttribute('fill', v.color || '#ff8800');
            cr.setAttribute('rx', '1.5');
            svg.appendChild(cr);
        });
        if (entryQueue.length > 8) {
            const more = this._text(queueX + 10, queueBaseY - 8 - 8 * 14 - 4,
                `+${entryQueue.length - 8}`,
                { fill: '#ffaa00', size: 7, font: 'Space Mono,monospace', anchor: 'middle' });
            svg.appendChild(more);
        }
    }

    // ── Ramp connector section ────────────────────────────────
    _buildConnector(upperLevel, lowerLevel) {
        const g = this._el('g');
        const y = this.levelOffsets[upperLevel.level_number] + CPR.FLOOR_H;

        // Background
        const bg = this._el('rect');
        bg.setAttribute('x', 0); bg.setAttribute('y', y);
        bg.setAttribute('width', CPR.SVG_W); bg.setAttribute('height', CPR.CONNECTOR_H);
        bg.setAttribute('fill', '#050510');
        g.appendChild(bg);

        // Ramp shaft (left side)
        const shaft = this._el('rect');
        shaft.setAttribute('x', CPR.MARGIN_L - 22); shaft.setAttribute('y', y);
        shaft.setAttribute('width', 64); shaft.setAttribute('height', CPR.CONNECTOR_H);
        shaft.setAttribute('fill', '#070a1a');
        shaft.setAttribute('stroke', '#1a2a50'); shaft.setAttribute('stroke-width', '1');
        g.appendChild(shaft);

        // Diagonal hatch lines in ramp shaft
        for (let hy = y; hy < y + CPR.CONNECTOR_H + 10; hy += 7) {
            const h = this._el('line');
            h.setAttribute('x1', CPR.MARGIN_L - 22); h.setAttribute('y1', hy);
            h.setAttribute('x2', CPR.MARGIN_L + 42); h.setAttribute('y2', hy + 7);
            h.setAttribute('stroke', 'rgba(68,136,255,0.12)');
            h.setAttribute('stroke-width', '0.8');
            g.appendChild(h);
        }

        // Arrow + label
        g.appendChild(this._text(CPR.MARGIN_L + 6, y + CPR.CONNECTOR_H / 2 + 4,
            '↕ RAMP', { fill: '#4488ff', size: 10, font: 'Space Mono,monospace' }));

        // Connector line (vehicle path guide, invisible but useful reference)
        const pathLine = this._el('line');
        pathLine.setAttribute('x1', CPR.MARGIN_L); pathLine.setAttribute('y1', y);
        pathLine.setAttribute('x2', CPR.MARGIN_L); pathLine.setAttribute('y2', y + CPR.CONNECTOR_H);
        pathLine.setAttribute('stroke', 'rgba(68,136,255,0.0)');
        g.appendChild(pathLine);

        return g;
    }

    // ── Vehicle path computation ──────────────────────────────
    computeEntryPath(vehicle, targetLevelNum, bayId, levels) {
        const targetLevel = levels.find(l => l.level_number === targetLevelNum);
        if (!targetLevel) return null;
        const bay = targetLevel.bays.find(b => b.id === bayId);
        if (!bay) return null;

        return this._buildPath(bay, targetLevelNum, levels, 'entry');
    }

    computeExitPath(vehicle, targetLevelNum, bayId, levels) {
        const targetLevel = levels.find(l => l.level_number === targetLevelNum);
        if (!targetLevel) return null;
        const bay = targetLevel.bays.find(b => b.id === bayId);
        if (!bay) return null;

        return this._buildPath(bay, targetLevelNum, levels, 'exit');
    }

    _buildPath(bay, targetLevelNum, levels, direction) {
        const baysPerRow = Math.max(...levels.find(l => l.level_number === targetLevelNum).bays.map(b => b.position));
        const bayW   = CPR.USABLE_W / baysPerRow;
        const bayX   = CPR.MARGIN_L + (bay.position - 1) * bayW + bayW / 2;
        const row    = bay.row;
        const fd     = CPR.ROW_FACES_DOWN[row];
        const aisleN = CPR.ROW_AISLE[row] || 1;

        const tFloorY = this.levelOffsets[targetLevelNum] || 0;
        const e1FloorY = this.levelOffsets[1] || 0;   // Level 1 is always entry level

        const tA1Y = tFloorY + CPR.AISLE1_CY;
        const tA2Y = tFloorY + CPR.AISLE2_CY;
        const e1A1Y = e1FloorY + CPR.AISLE1_CY;
        const aisleY = aisleN === 1 ? tA1Y : tA2Y;

        // Bay slot centre
        const baySlotY = tFloorY + CPR.ROW_Y[row] + (fd ? CPR.BAY_H * 0.28 : CPR.BAY_H * 0.72);

        const rightEntryX = CPR.SVG_W - CPR.MARGIN_R;
        const offscreenX  = CPR.SVG_W + 30;
        const rampX       = CPR.MARGIN_L;

        const entryPath = [
            { x: offscreenX,  y: e1A1Y },           // off-screen right
            { x: rightEntryX, y: e1A1Y },            // main entrance

            // Ramp chain if not on Level 1
            ...(targetLevelNum < 1 ? this._rampChain(targetLevelNum, levels) : []),

            // Navigate from left side to correct aisle on target floor
            ...(aisleN === 2
                ? [
                    { x: rampX,        y: tA1Y   },   // left end aisle 1
                    { x: rightEntryX,  y: tA1Y   },   // drive right to corridor
                    { x: rightEntryX,  y: tA2Y   },   // go down corridor
                    { x: bayX,         y: tA2Y   },   // drive to bay column
                  ]
                : [
                    { x: rampX,   y: tA1Y },           // left end aisle 1
                    { x: bayX,    y: tA1Y },            // drive to bay column
                  ]
            ),

            { x: bayX, y: baySlotY },                  // pull into bay
        ];

        if (direction === 'entry') return entryPath;

        // Exit path is the reverse minus the leading off-screen point
        const exit = [...entryPath].slice(1).reverse();
        const last = exit[exit.length - 1];
        exit.push({ x: offscreenX, y: last.y });
        return exit;
    }

    _rampChain(targetLevelNum, levels) {
        // Produce waypoints that travel down the left-side ramp from Level 1
        const sorted = [...levels].sort((a, b) => b.level_number - a.level_number);
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

    // ── SVG helpers ───────────────────────────────────────────
    _el(tag) { return document.createElementNS(this.NS, tag); }

    _text(x, y, content, opts = {}) {
        const t = this._el('text');
        t.setAttribute('x', x); t.setAttribute('y', y);
        t.setAttribute('fill', opts.fill || '#ff8800');
        t.setAttribute('font-size', opts.size || 10);
        t.setAttribute('font-family', opts.font || 'Space Mono,monospace');
        if (opts.weight)       t.setAttribute('font-weight', opts.weight);
        if (opts.anchor)       t.setAttribute('text-anchor', opts.anchor);
        if (opts.letterSpacing) t.setAttribute('letter-spacing', opts.letterSpacing);
        if (opts.ptrEvt)       t.setAttribute('pointer-events', opts.ptrEvt);
        t.textContent = content;
        return t;
    }

    _bindEvents(svg) {
        svg.querySelectorAll('.bay').forEach(el => {
            el.addEventListener('click', e => {
                e.stopPropagation();
                if (this._callbacks.bay) {
                    const bayId   = el.getAttribute('data-bay-id');
                    const lvlAttr = el.closest('g[id^="floor-g-"]')?.id.replace('floor-g-', '');
                    const lvlNum  = parseInt(lvlAttr ?? '0', 10);
                    this._callbacks.bay(e, bayId, lvlNum);
                }
            });
        });
        svg.querySelectorAll('.sv-vehicle').forEach(el => {
            el.addEventListener('click', e => {
                e.stopPropagation();
                if (this._callbacks.vehicle) {
                    this._callbacks.vehicle(e, el.getAttribute('data-vehicle-id'));
                }
            });
        });
    }
}

// ── VehicleAnimator ─────────────────────────────────────────────────
class VehicleAnimator {
    constructor(getAnimLayer) {
        this._getLayer = getAnimLayer;
        this.NS  = 'http://www.w3.org/2000/svg';
        this._active = new Map();   // vehicleId → animState
        this._rafId  = null;
        this._tick   = this._tick.bind(this);
        this._onNearMiss = null;
        this._nearMissCount = 0;
    }

    onNearMiss(fn) { this._onNearMiss = fn; }

    start(vehicle, path, onDone) {
        if (!path || path.length < 2) { if (onDone) onDone(); return; }
        this.remove(vehicle.id);    // clear any existing anim

        const layer = this._getLayer();
        if (!layer) return;

        // Car body
        const body = this._makeEl('rect', {
            width: 22, height: 12,
            fill: vehicle.color, rx: 2,
            class: 'av-body',
        });
        // Windshield
        const ws = this._makeEl('rect', {
            width: 12, height: 4,
            fill: 'rgba(255,255,255,0.32)', rx: 1,
            'pointer-events': 'none', class: 'av-ws',
        });
        // Glow
        const glow = this._makeEl('ellipse', {
            rx: 14, ry: 7,
            fill: vehicle.color,
            opacity: 0.15,
            'pointer-events': 'none',
        });

        layer.appendChild(glow);
        layer.appendChild(body);
        layer.appendChild(ws);

        const distOf = (i) => {
            const a = path[i], b = path[i + 1] || a;
            return Math.hypot(b.x - a.x, b.y - a.y);
        };
        const speedFor = dist => Math.max(0.008, Math.min(0.12, 55 / Math.max(dist, 1)));

        this._active.set(vehicle.id, {
            body, ws, glow, path,
            segIdx: 0, t: 0,
            speed: speedFor(distOf(0)),
            done: false, onDone
        });

        // Store original glow color for brake reset
        this._active.get(vehicle.id)._origGlowColor = vehicle.color;

        this._place(this._active.get(vehicle.id), path[0].x, path[0].y, 0);

        if (!this._rafId) this._rafId = requestAnimationFrame(this._tick);
    }

    _tick() {
        for (const [vid, a] of this._active.entries()) {
            if (a.done) {
                [a.body, a.ws, a.glow].forEach(e => e.parentNode?.removeChild(e));
                this._active.delete(vid);
                if (a.onDone) a.onDone();
                continue;
            }

            a.t += a.speed;

            if (a.t >= 1) {
                a.t = 0;
                a.segIdx++;
                if (a.segIdx >= a.path.length - 1) { a.done = true; continue; }

                const dist = Math.hypot(
                    a.path[a.segIdx + 1].x - a.path[a.segIdx].x,
                    a.path[a.segIdx + 1].y - a.path[a.segIdx].y
                );
                a.speed = Math.max(0.008, Math.min(0.12, 55 / Math.max(dist, 1)));
            }

            const from = a.path[a.segIdx];
            const to   = a.path[Math.min(a.segIdx + 1, a.path.length - 1)];
            const et   = this._ease(a.t);
            const x    = from.x + (to.x - from.x) * et;
            const y    = from.y + (to.y - from.y) * et;
            const angle = Math.atan2(to.y - from.y, to.x - from.x) * 180 / Math.PI;

            this._place(a, x, y, angle);
        }

        // ── Proximity / collision avoidance ──────────────────────
        const SAFE_DIST  = 28;   // SVG units — about 1.5 car lengths
        const BRAKE_DIST = 18;   // very close — near-miss
        const entries = [...this._active.entries()].filter(([,a]) => !a.done);

        for (let i = 0; i < entries.length; i++) {
            for (let j = i + 1; j < entries.length; j++) {
                const [, ai] = entries[i];
                const [, aj] = entries[j];
                if (ai.cx == null || aj.cx == null) continue;

                const dx   = ai.cx - aj.cx;
                const dy   = ai.cy - aj.cy;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < SAFE_DIST) {
                    // Slow the one with lower segment progress (the one behind)
                    const trailing = (ai.segIdx < aj.segIdx || (ai.segIdx === aj.segIdx && ai.t < aj.t)) ? ai : aj;
                    const maxSpeed = Math.max(0.003, (dist / SAFE_DIST) * 0.04);
                    trailing.speed = Math.min(trailing.speed, maxSpeed);

                    // Brake glow
                    if (!trailing._braking) {
                        trailing._braking = true;
                        trailing.glow.setAttribute('fill', '#ff2200');
                        trailing.glow.setAttribute('opacity', '0.4');
                    }

                    // Near-miss threshold
                    if (dist < BRAKE_DIST && !trailing._nearMissReported) {
                        trailing._nearMissReported = true;
                        this._nearMissCount = (this._nearMissCount || 0) + 1;
                        if (this._onNearMiss) this._onNearMiss();
                    }
                } else {
                    // Clear braking state
                    for (const [, a] of [entries[i], entries[j]]) {
                        if (a._braking) {
                            a._braking = false;
                            a.glow.setAttribute('fill', a._origGlowColor || a.body.getAttribute('fill'));
                            a.glow.setAttribute('opacity', '0.15');
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

    _place(a, x, y, angle) {
        a.cx = x; a.cy = y;
        const W = 22, H = 12;
        const tr = `rotate(${angle},${x},${y})`;
        a.body.setAttribute('x', x - W / 2); a.body.setAttribute('y', y - H / 2);
        a.body.setAttribute('transform', tr);
        a.ws.setAttribute('x',  x - 6);  a.ws.setAttribute('y',  y - 6);
        a.ws.setAttribute('transform', tr);
        a.glow.setAttribute('cx', x); a.glow.setAttribute('cy', y);
    }

    _makeEl(tag, attrs) {
        const el = document.createElementNS(this.NS, tag);
        Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
        return el;
    }

    _ease(t) { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; }

    remove(vehicleId) {
        const a = this._active.get(vehicleId);
        if (a) {
            [a.body, a.ws, a.glow].forEach(e => e.parentNode?.removeChild(e));
            this._active.delete(vehicleId);
        }
    }

    has(vehicleId) { return this._active.has(vehicleId); }

    clear() {
        for (const a of this._active.values()) {
            [a.body, a.ws, a.glow].forEach(e => e.parentNode?.removeChild(e));
        }
        this._active.clear();
        if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    }
}
