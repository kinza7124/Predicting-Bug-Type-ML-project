// Mobile Menu Toggle
const mobileMenuToggle = document.getElementById('mobileMenuToggle');
const mobileMenu = document.getElementById('mobileMenu');

// Set theme to dark mode permanently
document.body.setAttribute('data-theme', 'dark');

// Check if mobile device
function isMobileDevice() {
    return window.innerWidth <= 768 || /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

// Check if tablet device
function isTabletDevice() {
    return window.innerWidth > 768 && window.innerWidth <= 1024;
}

// Check if laptop device
function isLaptopDevice() {
    return window.innerWidth > 1024 && window.innerWidth <= 1440;
}

// Mobile Menu Toggle
function toggleMobileMenu() {
    mobileMenuToggle.classList.toggle('active');
    mobileMenu.classList.toggle('active');
    document.body.style.overflow = mobileMenu.classList.contains('active') ? 'hidden' : '';
}

mobileMenuToggle.addEventListener('click', toggleMobileMenu);

// Close mobile menu when clicking on a link
document.querySelectorAll('.mobile-nav-link').forEach(link => {
    link.addEventListener('click', () => {
        mobileMenuToggle.classList.remove('active');
        mobileMenu.classList.remove('active');
        document.body.style.overflow = '';
    });
});

// Close mobile menu when clicking outside
document.addEventListener('click', (e) => {
    if (!mobileMenu.contains(e.target) && !mobileMenuToggle.contains(e.target)) {
        mobileMenuToggle.classList.remove('active');
        mobileMenu.classList.remove('active');
        document.body.style.overflow = '';
    }
});

// Smooth Scrolling for Navigation Links
document.querySelectorAll('.nav-link, .mobile-nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        
        // Remove active class from all links
        document.querySelectorAll('.nav-link, .mobile-nav-link').forEach(l => l.classList.remove('active'));
        
        // Add active class to clicked link
        link.classList.add('active');
        
        // Get target section
        const targetId = link.getAttribute('href').substring(1);
        const targetSection = document.getElementById(targetId);
        
        if (targetSection) {
            // Close mobile menu if open
            mobileMenuToggle.classList.remove('active');
            mobileMenu.classList.remove('active');
            document.body.style.overflow = '';
            
            // Smooth scroll to target
            targetSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Update active navigation link on scroll
let scrollTimeout;
window.addEventListener('scroll', () => {
    // Throttle scroll events
    if (scrollTimeout) return;
    
    scrollTimeout = setTimeout(() => {
        const sections = document.querySelectorAll('.section, .hero');
        const navLinks = document.querySelectorAll('.nav-link, .mobile-nav-link');
        
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (window.pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
        
        scrollTimeout = null;
    }, 100);
});

// Initialize Charts with mobile optimization
function initializeCharts() {
    // Get colors from CSS variables for consistency
    const rootStyles = getComputedStyle(document.documentElement);
    const primaryColor = rootStyles.getPropertyValue('--primary').trim();
    const secondaryColor = rootStyles.getPropertyValue('--secondary').trim();
    const accentColor = rootStyles.getPropertyValue('--accent').trim();
    const successColor = rootStyles.getPropertyValue('--success').trim();
    const warningColor = rootStyles.getPropertyValue('--warning').trim();
    const errorColor = rootStyles.getPropertyValue('--error').trim();
    const textPrimary = rootStyles.getPropertyValue('--text-primary').trim();
    const borderColor = 'rgba(255,255,255,0.1)';
    
    // Check if mobile device
    const isMobile = isMobileDevice();
    const isTablet = isTabletDevice();
    const isLaptop = isLaptopDevice();
    const screenWidth = window.innerWidth;
    const containerPadding = isMobile ? 40 : isTablet ? 60 : 100;
    
    // Bug Type Distribution Chart - Fixed for mobile
    const bugTypeData = [{
        values: [6712, 2280, 1008],
        labels: ['Defect', 'Task', 'Enhancement'],
        type: 'pie',
        marker: {
            colors: [primaryColor, secondaryColor, accentColor]
        },
        textinfo: isMobile ? 'percent' : 'label+percent',
        textposition: isMobile ? 'inside' : 'outside',
        hoverinfo: 'label+percent+value',
        hovertemplate: '<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        hole: isMobile ? 0.2 : 0.3,
        rotation: 45,
        pull: isMobile ? 0.02 : 0.01,
        textfont: {
            size: isMobile ? 10 : 12,
            color: textPrimary
        },
        insidetextorientation: 'horizontal'
    }];
    
    const bugTypeLayout = {
        title: {
            text: '',  // Remove title from chart, will be handled by HTML
            font: { 
                size: isMobile ? 16 : isTablet ? 18 : 20, 
                color: textPrimary,
                family: 'Inter, sans-serif'
            }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { 
            color: textPrimary, 
            family: 'Inter, sans-serif',
            size: isMobile ? 10 : isTablet ? 11 : 12 
        },
        showlegend: !isMobile,
        legend: {
            orientation: 'h',
            y: isMobile ? -0.3 : isTablet ? -0.2 : -0.1,
            x: 0.5,
            xanchor: 'center',
            font: { 
                color: textPrimary, 
                size: isMobile ? 10 : isTablet ? 11 : 12,
                family: 'Inter, sans-serif'
            },
            bgcolor: 'rgba(0,0,0,0.3)',
            bordercolor: borderColor,
            borderwidth: 1
        },
        margin: isMobile ? { t: 20, b: 40, l: 20, r: 20 } : isTablet ? { t: 30, b: 50, l: 40, r: 40 } : { t: 30, b: 50, l: 50, r: 50 },
        height: isMobile ? 300 : isTablet ? 350 : 400,
        width: isMobile ? Math.min(screenWidth - containerPadding, 320) : undefined,
        annotations: isMobile ? [
            {
                font: {
                    size: 10,
                    color: textPrimary
                },
                showarrow: false,
                text: 'Touch slices for details',
                x: 0.5,
                y: -0.15,
                xref: 'paper',
                yref: 'paper'
            }
        ] : []
    };
    
    Plotly.newPlot('bugTypeChart', bugTypeData, bugTypeLayout, {
        responsive: true,
        displayModeBar: !isMobile,
        displaylogo: false,
        modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'select2d', 'lasso2d', 'toggleSpikelines'],
        scrollZoom: !isMobile
    });
    
    // Temporal Distribution Chart
    const temporalData = [{
        x: ['2023', '2024', '2025'],
        y: [3200, 4500, 2300],
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: primaryColor, width: isMobile ? 2 : 3 },
        marker: { color: secondaryColor, size: isMobile ? 6 : 8 },
        name: 'Bug Reports',
        fill: 'tozeroy',
        fillcolor: 'rgba(99, 102, 241, 0.1)'
    }];
    
    const temporalLayout = {
        title: {
            text: isMobile ? 'Bug Reports Over Time' : 'Bug Reports Over Time',
            font: { size: isMobile ? 12 : 16, color: textPrimary, family: 'Inter, sans-serif' },
            x: 0.5,
            xanchor: 'center'
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: textPrimary, size: isMobile ? 9 : 12, family: 'Inter, sans-serif' },
        xaxis: { 
            title: { text: isMobile ? '' : 'Year', font: { color: textPrimary, size: isMobile ? 9 : 12 } },
            gridcolor: borderColor,
            tickfont: { color: textPrimary, size: isMobile ? 9 : 12 },
            showgrid: !isMobile,
            fixedrange: isMobile
        },
        yaxis: { 
            title: { text: isMobile ? 'Bugs' : 'Number of Bugs', font: { color: textPrimary, size: isMobile ? 9 : 12 } },
            gridcolor: borderColor,
            tickfont: { color: textPrimary, size: isMobile ? 9 : 12 },
            showgrid: !isMobile,
            fixedrange: isMobile
        },
        hoverlabel: { font: { color: textPrimary, size: isMobile ? 10 : 12 } },
        margin: isMobile ? { t: 30, b: 30, l: 40, r: 20 } : isTablet ? { t: 40, b: 40, l: 50, r: 30 } : { t: 50, b: 50, l: 60, r: 50 },
        height: isMobile ? 250 : isTablet ? 300 : 350,
        width: isMobile ? Math.min(screenWidth - containerPadding, 300) : undefined,
        autosize: true
    };
    
    Plotly.newPlot('temporalChart', temporalData, temporalLayout, {
        responsive: true,
        displayModeBar: false,
        scrollZoom: false,
        doubleClick: false,
        showTips: false,
        staticPlot: isMobile
    });
    
    // Product Analysis Chart
    const productData = [{
        x: ['Firefox', 'Core', 'Toolkit', 'DevTools', 'WebExtensions'],
        y: [3500, 2800, 1500, 1200, 1000],
        type: 'bar',
        marker: {
            color: [primaryColor, secondaryColor, accentColor, successColor, warningColor]
        },
        text: isMobile ? [] : ['3500', '2800', '1500', '1200', '1000'],
        textposition: 'auto'
    }];
    
    const productLayout = {
        title: {
            text: isMobile ? 'Top Products' : 'Top Products by Bug Count',
            font: { size: isMobile ? 12 : 16, color: textPrimary, family: 'Inter, sans-serif' },
            x: 0.5,
            xanchor: 'center'
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: textPrimary, size: isMobile ? 9 : 12, family: 'Inter, sans-serif' },
        xaxis: { 
            title: { text: isMobile ? '' : 'Product', font: { color: textPrimary, size: isMobile ? 9 : 12 } },
            gridcolor: borderColor,
            tickfont: { color: textPrimary, size: isMobile ? 9 : 12 },
            tickangle: isMobile ? -45 : 0,
            showgrid: !isMobile,
            fixedrange: isMobile
        },
        yaxis: { 
            title: { text: isMobile ? 'Bugs' : 'Number of Bugs', font: { color: textPrimary, size: isMobile ? 9 : 12 } },
            gridcolor: borderColor,
            tickfont: { color: textPrimary, size: isMobile ? 9 : 12 },
            showgrid: !isMobile,
            fixedrange: isMobile
        },
        margin: isMobile ? { t: 30, b: 60, l: 40, r: 20 } : isTablet ? { t: 40, b: 80, l: 50, r: 30 } : { t: 50, b: 100, l: 60, r: 50 },
        height: isMobile ? 280 : isTablet ? 350 : 400,
        width: isMobile ? Math.min(screenWidth - containerPadding, 320) : undefined,
        autosize: true
    };
    
    Plotly.newPlot('productChart', productData, productLayout, {
        responsive: true,
        displayModeBar: false,
        scrollZoom: false,
        doubleClick: false,
        staticPlot: isMobile
    });
    
    // Before SMOTE Chart
    const beforeSmoteData = [{
        x: ['Defect', 'Task', 'Enhancement'],
        y: [6712, 2280, 1008],
        type: 'bar',
        marker: { color: errorColor },
        text: isMobile ? [] : ['6712', '2280', '1008'],
        textposition: 'auto'
    }];
    
    const beforeSmoteLayout = {
        title: {
            text: 'Before SMOTE (Imbalanced)',
            font: { size: isMobile ? 14 : 16, color: textPrimary }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: textPrimary, size: isMobile ? 10 : 12 },
        xaxis: { 
            gridcolor: borderColor,
            tickfont: { color: textPrimary, size: isMobile ? 10 : 12 }
        },
        yaxis: { 
            title: { text: 'Number of Samples', font: { color: textPrimary, size: isMobile ? 10 : 12 } },
            gridcolor: borderColor,
            tickfont: { color: textPrimary, size: isMobile ? 10 : 12 }
        },
        margin: isMobile ? { t: 40, b: 50, l: 60, r: 40 } : { t: 50, b: 50, l: 60, r: 50 }
    };
    
    Plotly.newPlot('beforeSmoteChart', beforeSmoteData, beforeSmoteLayout, {
        responsive: true,
        displayModeBar: !isMobile
    });
    
    // After SMOTE Chart
    const afterSmoteData = [{
        x: ['Defect', 'Task', 'Enhancement'],
        y: [6712, 6712, 6712],
        type: 'bar',
        marker: { color: successColor },
        text: isMobile ? [] : ['6712', '6712', '6712'],
        textposition: 'auto'
    }];
    
    const afterSmoteLayout = {
        title: {
            text: 'After SMOTE (Balanced)',
            font: { size: isMobile ? 14 : 16, color: textPrimary }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: textPrimary, size: isMobile ? 10 : 12 },
        xaxis: { 
            gridcolor: borderColor,
            tickfont: { color: textPrimary, size: isMobile ? 10 : 12 }
        },
        yaxis: { 
            title: { text: 'Number of Samples', font: { color: textPrimary, size: isMobile ? 10 : 12 } },
            gridcolor: borderColor,
            tickfont: { color: textPrimary, size: isMobile ? 10 : 12 }
        },
        margin: isMobile ? { t: 40, b: 50, l: 60, r: 40 } : { t: 50, b: 50, l: 60, r: 50 }
    };
    
    Plotly.newPlot('afterSmoteChart', afterSmoteData, afterSmoteLayout, {
        responsive: true,
        displayModeBar: !isMobile
    });
    
    // Models Comparison Chart
    const modelsData = [{
        y: ['Perceptron', 'Gaussian NB', 'Gradient Boosting', 'MLP Neural Network', 'Logistic Regression', 'SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Stacking Classifier'],
        x: [33.33, 62.51, 78.40, 75.62, 80.98, 81.53, 82.70, 83.81, 91.73, 94.34],
        type: 'bar',
        orientation: 'h',
        marker: {
            color: [33.33, 62.51, 78.40, 75.62, 80.98, 81.53, 82.70, 83.81, 91.73, 94.34],
            colorscale: 'Viridis',
            showscale: false
        },
        text: isMobile ? [] : ['33.33%', '62.51%', '78.40%', '75.62%', '80.98%', '81.53%', '82.70%', '83.81%', '91.73%', '94.34%'],
        textposition: 'auto'
    }];
    
    const modelsLayout = {
        title: {
            text: 'Model Accuracy Comparison',
            font: { size: isMobile ? 14 : 18, color: textPrimary }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: textPrimary, size: isMobile ? 10 : 12 },
        xaxis: { 
            title: { text: 'Accuracy (%)', font: { color: textPrimary, size: isMobile ? 10 : 12 } },
            gridcolor: borderColor,
            tickfont: { color: textPrimary, size: isMobile ? 10 : 12 },
            range: [0, 100]
        },
        yaxis: { 
            title: { text: 'Model', font: { color: textPrimary, size: isMobile ? 10 : 12 } },
            gridcolor: borderColor,
            tickfont: { color: textPrimary, size: isMobile ? 8 : 12 },
            automargin: true
        },
        margin: isMobile ? { t: 40, b: 50, l: 180, r: 40 } : { t: 50, b: 50, l: 180, r: 50 },
        height: isMobile ? 500 : 600
    };
    
    Plotly.newPlot('modelsChart', modelsData, modelsLayout, {
        responsive: true,
        displayModeBar: !isMobile
    });
    
    // Confusion Matrix
    const confusionData = [{
        z: [[1320, 45, 25], [35, 1340, 35], [20, 30, 1350]],
        x: ['Defect', 'Task', 'Enhancement'],
        y: ['Defect', 'Task', 'Enhancement'],
        type: 'heatmap',
        colorscale: [
            [0, 'rgba(99, 102, 241, 0.1)'],
            [0.5, 'rgba(99, 102, 241, 0.5)'],
            [1, 'rgba(99, 102, 241, 1)']
        ],
        showscale: !isMobile,
        hoverongaps: false,
        text: isMobile ? [] : [['1320', '45', '25'], ['35', '1340', '35'], ['20', '30', '1350']],
        texttemplate: isMobile ? '' : '%{text}',
        textfont: { color: textPrimary, size: isMobile ? 8 : 12 }
    }];
    
    const confusionLayout = {
        title: {
            text: 'Confusion Matrix - Stacking Classifier',
            font: { size: isMobile ? 14 : 16, color: textPrimary }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: textPrimary, size: isMobile ? 10 : 12 },
        xaxis: { 
            title: { text: 'Predicted', font: { color: textPrimary, size: isMobile ? 10 : 12 } },
            tickfont: { color: textPrimary, size: isMobile ? 10 : 12 }
        },
        yaxis: { 
            title: { text: 'Actual', font: { color: textPrimary, size: isMobile ? 10 : 12 } },
            tickfont: { color: textPrimary, size: isMobile ? 10 : 12 }
        },
        margin: isMobile ? { t: 40, b: 50, l: 60, r: 40 } : { t: 50, b: 50, l: 60, r: 50 },
        height: isMobile ? 300 : 400
    };
    
    Plotly.newPlot('confusionMatrix', confusionData, confusionLayout, {
        responsive: true,
        displayModeBar: !isMobile
    });
    
    // Add touch events for bug type chart on mobile
    if (isMobile) {
        const bugTypeChart = document.getElementById('bugTypeChart');
        bugTypeChart.addEventListener('click', function(eventData) {
            // Highlight the clicked segment
            if (eventData.points && eventData.points[0]) {
                const point = eventData.points[0];
                // Create a simple alert or tooltip with bug type info
                const bugInfo = document.createElement('div');
                bugInfo.className = 'mobile-bug-info';
                bugInfo.innerHTML = `
                    <div class="bug-info-content">
                        <h4>${point.label}</h4>
                        <p>Count: ${point.value.toLocaleString()}</p>
                        <p>Percentage: ${(point.percent * 100).toFixed(1)}%</p>
                    </div>
                `;
                bugInfo.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: rgba(15, 15, 35, 0.95);
                    backdrop-filter: blur(20px);
                    border: 1px solid ${borderColor};
                    border-radius: 10px;
                    padding: 20px;
                    z-index: 10000;
                    max-width: 80%;
                    text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                `;
                
                document.body.appendChild(bugInfo);
                
                // Remove after 2 seconds
                setTimeout(() => {
                    if (bugInfo.parentNode) {
                        bugInfo.parentNode.removeChild(bugInfo);
                    }
                }, 2000);
            }
        });
    }
}

// Add CSS for mobile bug info
const style = document.createElement('style');
style.textContent = `
    .mobile-bug-info .bug-info-content h4 {
        color: var(--text-primary);
        margin-bottom: 10px;
        font-size: 18px;
    }
    
    .mobile-bug-info .bug-info-content p {
        color: var(--text-secondary);
        margin: 5px 0;
        font-size: 14px;
    }
    
    @media (max-width: 768px) {
        .floating-card {
            height: 400px !important;
            overflow: hidden;
        }
        
        #bugTypeChart {
            min-height: 350px;
            touch-action: manipulation;
        }
        
        .js-plotly-plot .plotly .modebar {
            display: none !important;
        }
    }
`;
document.head.appendChild(style);

// Update charts on window resize
let resizeTimeout;
window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(function() {
        // Clear existing plots
        Plotly.purge('bugTypeChart');
        Plotly.purge('temporalChart');
        Plotly.purge('productChart');
        Plotly.purge('beforeSmoteChart');
        Plotly.purge('afterSmoteChart');
        Plotly.purge('modelsChart');
        Plotly.purge('confusionMatrix');
        
        // Reinitialize charts with new dimensions
        initializeCharts();
    }, 250);
});

// Intersection Observer for Animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all cards and sections for animation
document.addEventListener('DOMContentLoaded', () => {
    // Initialize charts first
    initializeCharts();
    
    // Then setup animations
    const animatedElements = document.querySelectorAll('.stat-card, .overview-card, .data-card, .chart-card, .pipeline-step, .model-card, .insight-card, .performance-card, .comparison-card, .dataset-card');
    
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
    
    // Animate section titles
    const sectionTitles = document.querySelectorAll('.section-title');
    sectionTitles.forEach(title => {
        title.style.opacity = '0';
        title.style.transform = 'translateY(30px)';
        title.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(title);
    });
});

// Add loading animation
window.addEventListener('load', () => {
    document.body.classList.add('loaded');
    
    // Add a slight delay to ensure all elements are loaded
    setTimeout(() => {
        const preloader = document.createElement('div');
        preloader.style.position = 'fixed';
        preloader.style.top = '0';
        preloader.style.left = '0';
        preloader.style.width = '100%';
        preloader.style.height = '100%';
        preloader.style.background = 'var(--bg-primary)';
        preloader.style.zIndex = '9999';
        preloader.style.opacity = '1';
        preloader.style.transition = 'opacity 0.5s ease';
        preloader.style.pointerEvents = 'none';
        preloader.id = 'preloader';
        document.body.appendChild(preloader);
        
        // Fade out preloader
        setTimeout(() => {
            preloader.style.opacity = '0';
            setTimeout(() => {
                if (preloader.parentNode) {
                    preloader.parentNode.removeChild(preloader);
                }
            }, 500);
        }, 100);
    }, 100);
});

// Navbar background on scroll
let scrollTimeoutNavbar;
window.addEventListener('scroll', () => {
    if (scrollTimeoutNavbar) return;
    
    scrollTimeoutNavbar = setTimeout(() => {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(15, 15, 35, 0.98)';
            navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
            navbar.style.backdropFilter = 'blur(20px)';
        } else {
            navbar.style.background = 'rgba(15, 15, 35, 0.95)';
            navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.1)';
            navbar.style.backdropFilter = 'blur(20px)';
        }
        
        scrollTimeoutNavbar = null;
    }, 100);
});

// Add scroll progress indicator
let scrollProgressTimeout;
window.addEventListener('scroll', () => {
    if (scrollProgressTimeout) return;
    
    scrollProgressTimeout = setTimeout(() => {
        let scrollProgress = document.getElementById('scroll-progress');
        if (!scrollProgress) {
            scrollProgress = document.createElement('div');
            scrollProgress.id = 'scroll-progress';
            document.body.appendChild(scrollProgress);
        }
        
        const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (winScroll / height) * 100;
        scrollProgress.style.width = scrolled + '%';
        
        scrollProgressTimeout = null;
    }, 50);
});

// Add click effect for mobile menu links
const mobileNavLinks = document.querySelectorAll('.mobile-nav-link');
mobileNavLinks.forEach(link => {
    link.addEventListener('click', () => {
        mobileNavLinks.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
    });
});

// Handle orientation change
let orientationTimeout;
window.addEventListener('orientationchange', () => {
    clearTimeout(orientationTimeout);
    orientationTimeout = setTimeout(() => {
        // Close mobile menu on orientation change
        mobileMenuToggle.classList.remove('active');
        mobileMenu.classList.remove('active');
        document.body.style.overflow = '';
        
        // Clear and reinitialize charts
        Plotly.purge('bugTypeChart');
        Plotly.purge('temporalChart');
        Plotly.purge('productChart');
        Plotly.purge('beforeSmoteChart');
        Plotly.purge('afterSmoteChart');
        Plotly.purge('modelsChart');
        Plotly.purge('confusionMatrix');
        
        // Reinitialize charts
        setTimeout(() => {
            initializeCharts();
        }, 300);
    }, 300);
});

// Prevent zoom on double-tap (iOS)
let lastTouchEnd = 0;
document.addEventListener('touchend', (event) => {
    const now = (new Date()).getTime();
    if (now - lastTouchEnd <= 300) {
        event.preventDefault();
    }
    lastTouchEnd = now;
}, false);

// Handle touch events for charts
document.addEventListener('touchstart', function(e) {
    // Allow touch scrolling on charts
    if (e.target.closest('.js-plotly-plot')) {
        e.stopPropagation();
    }
}, { passive: true });

// Initialize everything when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCharts);
} else {
    initializeCharts();
}
// Debug function for laptop screens
function debugLaptopLayout() {
    const screenWidth = window.innerWidth;
    const dataGrid = document.querySelector('.data-grid');
    const dataCards = document.querySelectorAll('.data-card, .chart-card');
    
    console.log('Screen width:', screenWidth);
    console.log('Data grid element:', dataGrid);
    console.log('Data grid computed style:', window.getComputedStyle(dataGrid));
    console.log('Number of data cards:', dataCards.length);
    
    dataCards.forEach((card, index) => {
        console.log(`Card ${index}:`, card);
        console.log(`Card ${index} computed style:`, window.getComputedStyle(card));
    });
    
    // Force visibility if needed
    if (screenWidth >= 769) {
        if (dataGrid) {
            dataGrid.style.display = 'grid';
            dataGrid.style.gridTemplateColumns = '1fr 1fr';
            dataGrid.style.gap = '2rem';
            dataGrid.style.visibility = 'visible';
            dataGrid.style.opacity = '1';
        }
        
        dataCards.forEach(card => {
            card.style.display = 'block';
            card.style.visibility = 'visible';
            card.style.opacity = '1';
        });
    }
}

// Run debug function on load and resize
window.addEventListener('load', debugLaptopLayout);
window.addEventListener('resize', debugLaptopLayout);

// Force layout update after DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(debugLaptopLayout, 1000);
});
