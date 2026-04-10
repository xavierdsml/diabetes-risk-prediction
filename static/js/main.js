/* ===================================================================
   DiabetesPredict — Main JavaScript (Lark Finserv-inspired)
   =================================================================== */

// ---------- AOS Init + DOM Ready ----------
document.addEventListener('DOMContentLoaded', function () {
    // Initialize AOS scroll animations
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 700,
            easing: 'ease-out-cubic',
            once: true,
            offset: 60
        });
    }

    // Scroll to result section after prediction
    const resultSection = document.getElementById('resultSection');
    if (resultSection) {
        setTimeout(function() {
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 200);
    }

    // Navbar background change on scroll
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', function () {
            if (window.scrollY > 20) {
                navbar.style.boxShadow = '0 4px 20px rgba(0,0,0,0.08)';
            } else {
                navbar.style.boxShadow = 'none';
            }
        });
    }
});

// ---------- Form validation ----------
(function () {
    'use strict';
    const form = document.getElementById('predictForm');
    if (!form) return;
    form.addEventListener('submit', function (event) {
        if (!form.checkValidity()) {
            event.preventDefault();
            event.stopPropagation();
        }
        form.classList.add('was-validated');
    });
})();

// ---------- Dashboard chart rendering ----------
function renderDashboardCharts(metrics) {
    if (!metrics || Object.keys(metrics).length === 0) return;

    const models = Object.keys(metrics);
    const accs = models.map(m => metrics[m].accuracy * 100);
    const precs = models.map(m => metrics[m].precision * 100);
    const recalls = models.map(m => metrics[m].recall * 100);
    const f1s = models.map(m => metrics[m].f1 * 100);
    const rocs = models.map(m => metrics[m].roc_auc * 100);

    // Gradient-inspired color palette (blue to purple)
    const colors = [
        '#2563eb', '#3b82f6', '#6366f1', '#7c3aed',
        '#8b5cf6', '#a855f7', '#0ea5e9'
    ];

    const baseLayout = {
        margin: { t: 30, b: 100, l: 60, r: 20 },
        font: { family: 'Inter, system-ui, sans-serif', size: 12 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            tickangle: -25,
            tickfont: { size: 11, color: '#64748b' },
            gridcolor: 'rgba(0,0,0,0.04)'
        },
        yaxis: {
            tickfont: { size: 11, color: '#64748b' },
            gridcolor: 'rgba(0,0,0,0.06)'
        },
    };

    const config = { responsive: true, displayModeBar: false };

    // 1. Accuracy bar chart
    if (document.getElementById('accuracyChart')) {
        Plotly.newPlot('accuracyChart', [{
            x: models,
            y: accs,
            type: 'bar',
            marker: {
                color: colors,
                line: { width: 0 },
                cornerradius: 6
            },
            text: accs.map(v => v.toFixed(2) + '%'),
            textposition: 'outside',
            textfont: { size: 11, color: '#1e293b', family: 'Inter' },
            hovertemplate: '<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>'
        }], Object.assign({}, baseLayout, {
            yaxis: Object.assign({}, baseLayout.yaxis, { title: { text: 'Accuracy (%)', font: { size: 12, color: '#64748b' } }, range: [0, 108] }),
        }), config);
    }

    // 2. Precision / Recall / F1 grouped bar chart
    if (document.getElementById('prfChart')) {
        Plotly.newPlot('prfChart', [
            {
                x: models, y: precs, name: 'Precision',
                type: 'bar', marker: { color: '#2563eb', cornerradius: 4 },
                hovertemplate: '<b>%{x}</b><br>Precision: %{y:.2f}%<extra></extra>'
            },
            {
                x: models, y: recalls, name: 'Recall',
                type: 'bar', marker: { color: '#7c3aed', cornerradius: 4 },
                hovertemplate: '<b>%{x}</b><br>Recall: %{y:.2f}%<extra></extra>'
            },
            {
                x: models, y: f1s, name: 'F1 Score',
                type: 'bar', marker: { color: '#0ea5e9', cornerradius: 4 },
                hovertemplate: '<b>%{x}</b><br>F1: %{y:.2f}%<extra></extra>'
            }
        ], Object.assign({}, baseLayout, {
            barmode: 'group',
            yaxis: Object.assign({}, baseLayout.yaxis, { title: { text: 'Score (%)', font: { size: 12, color: '#64748b' } }, range: [0, 108] }),
            legend: { orientation: 'h', y: 1.15, font: { size: 11, color: '#64748b' } }
        }), config);
    }

    // 3. ROC-AUC bar chart
    if (document.getElementById('rocChart')) {
        Plotly.newPlot('rocChart', [{
            x: models,
            y: rocs,
            type: 'bar',
            marker: {
                color: rocs.map(v => v >= 83 ? '#22c55e' : v >= 78 ? '#2563eb' : '#f59e0b'),
                line: { width: 0 },
                cornerradius: 6
            },
            text: rocs.map(v => v.toFixed(2) + '%'),
            textposition: 'outside',
            textfont: { size: 11, color: '#1e293b', family: 'Inter' },
            hovertemplate: '<b>%{x}</b><br>ROC-AUC: %{y:.2f}%<extra></extra>'
        }], Object.assign({}, baseLayout, {
            yaxis: Object.assign({}, baseLayout.yaxis, { title: { text: 'ROC-AUC (%)', font: { size: 12, color: '#64748b' } }, range: [0, 108] }),
        }), config);
    }
}
