// Bug Classification Study — interactions and charts

(function () {
    'use strict';

    // ----------------------------------------------------------------------
    // Navigation
    // ----------------------------------------------------------------------

    const masthead = document.getElementById('masthead');
    const navToggle = document.getElementById('navToggle');
    const drawer = document.getElementById('drawer');
    const navLinks = document.querySelectorAll('.nav-link');

    function setDrawer(open) {
        drawer.classList.toggle('is-open', open);
        navToggle.setAttribute('aria-expanded', String(open));
        navToggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
        document.body.classList.toggle('drawer-open', open);
    }

    navToggle.addEventListener('click', () => setDrawer(!drawer.classList.contains('is-open')));
    drawer.querySelectorAll('a').forEach(link => link.addEventListener('click', () => setDrawer(false)));
    document.addEventListener('keydown', e => { if (e.key === 'Escape') setDrawer(false); });
    window.addEventListener('resize', () => { if (window.innerWidth > 860) setDrawer(false); });

    // Hairline under the masthead once the page scrolls
    const onScroll = () => masthead.classList.toggle('is-scrolled', window.scrollY > 8);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    // Highlight the section currently in view
    const tracked = ['overview', 'data', 'preprocessing', 'models', 'performance', 'insights']
        .map(id => document.getElementById(id))
        .filter(Boolean);

    const sectionObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const id = entry.target.id;
            navLinks.forEach(link => link.classList.toggle('is-active', link.getAttribute('href') === '#' + id));
        });
    }, { rootMargin: '-45% 0px -50% 0px' });

    tracked.forEach(section => sectionObserver.observe(section));

    // ----------------------------------------------------------------------
    // Reveal on scroll
    // ----------------------------------------------------------------------

    const revealObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                revealObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

    document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

    // ----------------------------------------------------------------------
    // Charts
    // ----------------------------------------------------------------------

    if (typeof Plotly === 'undefined') return;

    const css = getComputedStyle(document.documentElement);
    const token = name => css.getPropertyValue(name).trim();

    const C = {
        ink: token('--ink'),
        inkSoft: token('--ink-soft'),
        inkMuted: token('--ink-muted'),
        rule: token('--rule-soft'),
        brass: token('--brass'),
        brassLight: token('--brass-light'),
        oxblood: token('--oxblood'),
        sage: token('--sage'),
        paper: token('--paper'),
        paperDeep: token('--paper-deep'),
        stone: '#b5aa96',
        ivory: token('--ivory'),
        ivoryMuted: token('--ivory-muted'),
        nightRule: token('--night-rule')
    };

    const SANS = 'Jost, Helvetica Neue, Arial, sans-serif';
    const SERIF = 'Cormorant Garamond, Georgia, serif';

    const isSmall = () => window.innerWidth <= 560;

    const config = {
        responsive: true,
        displayModeBar: false,
        displaylogo: false,
        scrollZoom: false,
        doubleClick: false
    };

    function baseLayout(overrides) {
        const small = isSmall();
        return Object.assign({
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: SANS, color: C.inkSoft, size: small ? 11 : 12 },
            margin: { t: 16, r: 8, b: 44, l: 52 },
            height: small ? 280 : 320,
            showlegend: false,
            bargap: 0.45,
            hoverlabel: {
                bgcolor: C.ink,
                bordercolor: C.ink,
                font: { family: SANS, color: C.paper, size: 12 }
            },
            xaxis: {
                showgrid: false,
                zeroline: false,
                linecolor: C.ink,
                linewidth: 1,
                ticks: '',
                tickfont: { color: C.inkSoft },
                fixedrange: true
            },
            yaxis: {
                gridcolor: C.rule,
                gridwidth: 1,
                zeroline: false,
                showline: false,
                ticks: '',
                tickfont: { color: C.inkMuted },
                separatethousands: true,
                tickformat: ',d',
                fixedrange: true
            }
        }, overrides);
    }

    function renderCharts() {
        const small = isSmall();

        // Fig. 1 — type distribution
        Plotly.react('bugTypeChart', [{
            type: 'pie',
            values: [6712, 2280, 1008],
            labels: ['Defect', 'Task', 'Enhancement'],
            hole: 0.72,
            sort: false,
            direction: 'clockwise',
            rotation: 0,
            marker: {
                colors: [C.ink, C.brass, C.stone],
                line: { color: C.paper, width: 3 }
            },
            textinfo: 'none',
            hovertemplate: '<b>%{label}</b><br>%{value:,} reports · %{percent}<extra></extra>'
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { family: SANS, color: C.inkSoft, size: 12 },
            margin: { t: 8, r: 8, b: 8, l: 8 },
            height: small ? 300 : 360,
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.04,
                yanchor: 'top',
                font: { family: SANS, size: 12, color: C.inkSoft },
                itemclick: false,
                itemdoubleclick: false
            },
            hoverlabel: { bgcolor: C.ink, bordercolor: C.ink, font: { family: SANS, color: C.paper } },
            annotations: [{
                text: '67%',
                showarrow: false,
                font: { family: SERIF, size: small ? 38 : 48, color: C.ink },
                x: 0.5, y: 0.56, xref: 'paper', yref: 'paper', yanchor: 'middle'
            }, {
                text: 'DEFECTS',
                showarrow: false,
                font: { family: SANS, size: 11, color: C.inkMuted },
                x: 0.5, y: 0.40, xref: 'paper', yref: 'paper', yanchor: 'middle'
            }]
        }, config);

        // Fig. 2 — reports per year
        Plotly.react('temporalChart', [{
            type: 'scatter',
            mode: 'lines+markers+text',
            x: ['2023', '2024', '2025'],
            y: [3200, 4500, 2300],
            line: { color: C.ink, width: 1.5, shape: 'linear' },
            marker: { color: C.paper, size: 9, line: { color: C.ink, width: 1.5 } },
            fill: 'tozeroy',
            fillcolor: 'rgba(154, 120, 64, 0.10)',
            text: ['3,200', '4,500', '2,300'],
            textposition: 'top center',
            textfont: { family: SERIF, size: 17, color: C.ink },
            cliponaxis: false,
            hovertemplate: '%{x}: %{y:,} reports<extra></extra>'
        }], baseLayout({
            margin: { t: 24, r: 32, b: 44, l: 52 },
            xaxis: Object.assign(baseLayout().xaxis, { type: 'category' }),
            yaxis: Object.assign(baseLayout().yaxis, { range: [0, 5400] })
        }), config);

        // Fig. 3 — product counts
        const products = ['Firefox', 'Core', 'Toolkit', 'DevTools', 'WebExtensions'];
        const productCounts = [3500, 2800, 1500, 1200, 1000];
        Plotly.react('productChart', [{
            type: 'bar',
            x: products,
            y: productCounts,
            marker: { color: products.map((_, i) => i === 0 ? C.brass : C.ink) },
            text: productCounts.map(v => v.toLocaleString()),
            textposition: 'outside',
            textfont: { family: SERIF, size: 16, color: C.ink },
            cliponaxis: false,
            hovertemplate: '%{x}: %{y:,} reports<extra></extra>'
        }], baseLayout({
            margin: { t: 24, r: 8, b: small ? 64 : 44, l: 52 },
            xaxis: Object.assign(baseLayout().xaxis, { tickangle: small ? -35 : 0 }),
            yaxis: Object.assign(baseLayout().yaxis, { range: [0, 4100] })
        }), config);

        // Fig. 4 / 5 — SMOTE before and after
        const classes = ['Defect', 'Task', 'Enhancement'];
        const smoteLayout = baseLayout({
            margin: { t: 24, r: 8, b: 44, l: 52 },
            yaxis: Object.assign(baseLayout().yaxis, { range: [0, 7800] })
        });

        Plotly.react('beforeSmoteChart', [{
            type: 'bar',
            x: classes,
            y: [6712, 2280, 1008],
            marker: { color: [C.ink, C.stone, C.stone] },
            text: ['6,712', '2,280', '1,008'],
            textposition: 'outside',
            textfont: { family: SERIF, size: 16, color: C.ink },
            cliponaxis: false,
            hovertemplate: '%{x}: %{y:,} samples<extra></extra>'
        }], smoteLayout, config);

        Plotly.react('afterSmoteChart', [{
            type: 'bar',
            x: classes,
            y: [6712, 6712, 6712],
            marker: { color: [C.ink, C.brass, C.brass] },
            text: ['6,712', '6,712', '6,712'],
            textposition: 'outside',
            textfont: { family: SERIF, size: 16, color: C.ink },
            cliponaxis: false,
            hovertemplate: '%{x}: %{y:,} samples<extra></extra>'
        }], smoteLayout, config);

        // Fig. 6 — model comparison (dark section), sorted ascending so the best sits on top
        const models = [
            ['Perceptron', 33.33],
            ['Gaussian NB', 62.51],
            ['MLP Neural Network', 75.62],
            ['Gradient Boosting', 78.40],
            ['Logistic Regression', 80.98],
            ['SVM', 81.53],
            ['KNN', 82.70],
            ['Decision Tree', 83.81],
            ['Random Forest', 91.73],
            ['Stacking Classifier', 94.34]
        ];
        const best = models.length - 1;

        Plotly.react('modelsChart', [{
            type: 'bar',
            orientation: 'h',
            y: models.map(m => m[0]),
            x: models.map(m => m[1]),
            marker: { color: models.map((_, i) => i === best ? C.brassLight : 'rgba(239, 233, 220, 0.82)') },
            text: models.map(m => m[1].toFixed(2) + '%'),
            textposition: 'outside',
            textfont: { family: SERIF, size: small ? 14 : 17, color: C.ivory },
            cliponaxis: false,
            hovertemplate: '%{y}: %{x:.2f}%<extra></extra>'
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: SANS, color: C.ivoryMuted, size: small ? 11 : 13 },
            margin: { t: 8, r: small ? 56 : 72, b: 40, l: small ? 128 : 170 },
            height: small ? 440 : 520,
            bargap: 0.5,
            showlegend: false,
            hoverlabel: { bgcolor: C.ivory, bordercolor: C.ivory, font: { family: SANS, color: C.ink } },
            xaxis: {
                range: [0, 100],
                ticksuffix: '%',
                gridcolor: C.nightRule,
                zeroline: false,
                showline: false,
                ticks: '',
                tickfont: { color: C.ivoryMuted },
                fixedrange: true
            },
            yaxis: {
                showgrid: false,
                ticks: '',
                ticklabelstandoff: 12,
                tickfont: { color: C.ivory },
                fixedrange: true
            }
        }, config);

        // Fig. 7 — confusion matrix
        const z = [[1320, 45, 25], [35, 1340, 35], [20, 30, 1350]];
        Plotly.react('confusionMatrix', [{
            type: 'heatmap',
            z: z,
            x: classes,
            y: classes,
            colorscale: [
                [0, C.paperDeep],
                [0.08, '#d9c9a6'],
                [1, C.ink]
            ],
            showscale: false,
            xgap: 3,
            ygap: 3,
            hovertemplate: 'Actual %{y} → predicted %{x}<br>%{z:,} reports<extra></extra>'
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: SANS, color: C.inkSoft, size: small ? 11 : 12 },
            margin: { t: 8, r: 8, b: 56, l: small ? 96 : 110 },
            height: small ? 320 : 380,
            hoverlabel: { bgcolor: C.ink, bordercolor: C.ink, font: { family: SANS, color: C.paper } },
            xaxis: {
                title: { text: 'PREDICTED', font: { size: 11, color: C.inkMuted }, standoff: 14 },
                side: 'bottom',
                ticks: '',
                showgrid: false,
                fixedrange: true
            },
            yaxis: {
                title: { text: 'ACTUAL', font: { size: 11, color: C.inkMuted }, standoff: 14 },
                autorange: 'reversed',
                ticks: '',
                showgrid: false,
                fixedrange: true
            },
            annotations: z.flatMap((row, r) => row.map((v, c) => ({
                x: classes[c],
                y: classes[r],
                text: v.toLocaleString(),
                showarrow: false,
                font: { family: SERIF, size: small ? 17 : 22, color: r === c ? C.paper : C.ink }
            })))
        }, config);
    }

    renderCharts();

    // Re-render only when crossing the small-screen breakpoint; Plotly handles width itself
    let wasSmall = isSmall();
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            if (isSmall() !== wasSmall) {
                wasSmall = isSmall();
                renderCharts();
            }
        }, 200);
    });
})();
