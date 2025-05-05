document.addEventListener('DOMContentLoaded', function() {
    // Section navigation
    const navLinks = document.querySelectorAll('.admin-nav .nav-link');
    const sections = document.querySelectorAll('.admin-section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const sectionId = this.getAttribute('data-section');
            
            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Show selected section
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === sectionId) {
                    section.classList.add('active');
                }
            });
            
            // Load section data if needed
            loadSectionData(sectionId);
        });
    });
    
    // Load data for the currently active section
    const activeSection = document.querySelector('.admin-section.active');
    if (activeSection) {
        loadSectionData(activeSection.id);
    }
    
    // Search functionality
    const searchInput = document.querySelector('.admin-search input');
    if (searchInput) {
        searchInput.addEventListener('keyup', function(e) {
            if (e.key === 'Enter') {
                performSearch(this.value);
            }
        });
        
        const searchButton = document.querySelector('.admin-search button');
        if (searchButton) {
            searchButton.addEventListener('click', function() {
                performSearch(searchInput.value);
            });
        }
    }
    
    // Initialize any charts if needed
    initializeCharts();
});

// Load data for each section
function loadSectionData(sectionId) {
    switch (sectionId) {
        case 'dashboard':
            loadDashboardData();
            break;
        case 'analytics':
            loadAnalyticsData();
            break;
        case 'users':
            loadUsersData();
            break;
        case 'api-keys':
            loadApiKeysData();
            break;
        case 'settings':
            loadSettingsData();
            break;
        case 'knowledge':
            loadKnowledgeData();
            break;
        case 'logs':
            loadLogsData();
            break;
    }
}

// Dashboard data
function loadDashboardData() {
    fetchData('/api/admin/analytics', (data) => {
        if (data && data.success) {
            updateDashboardStats(data.data);
            updateDashboardCharts(data.data);
        }
    });
}

// Update dashboard statistics
function updateDashboardStats(data) {
    const statsContainer = document.querySelector('.dashboard-stats');
    if (!statsContainer) return;
    
    // Sample implementation - replace with actual data
    if (data) {
        statsContainer.innerHTML = `
            <div class="stat-card">
                <div class="stat-card-title">Total Requests</div>
                <div class="stat-card-value">${data.total_requests || 0}</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-title">Unique Users</div>
                <div class="stat-card-value">${data.unique_users || 0}</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-title">Avg Response Time</div>
                <div class="stat-card-value">${data.avg_response_time ? data.avg_response_time.toFixed(2) + 's' : '0s'}</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-title">Error Rate</div>
                <div class="stat-card-value">${data.error_rate ? (data.error_rate * 100).toFixed(2) + '%' : '0%'}</div>
            </div>
        `;
    }
}

// Update dashboard charts
function updateDashboardCharts(data) {
    // This would be implemented with a charting library like Chart.js
    console.log('Dashboard charts would be updated with:', data);
}

// Analytics data
function loadAnalyticsData() {
    fetchData('/api/admin/analytics?detailed=true', (data) => {
        if (data && data.success) {
            // Update analytics UI
            console.log('Analytics data loaded:', data);
        }
    });
}

// Users data
function loadUsersData() {
    fetchData('/api/admin/users', (data) => {
        if (data && data.users) {
            // Update users UI
            console.log('Users data loaded:', data);
        }
    });
}

// API Keys data
function loadApiKeysData() {
    fetchData('/api/admin/api-keys', (data) => {
        if (data && data.api_keys) {
            // Update API keys UI
            console.log('API keys loaded:', data);
        }
    });
}

// Settings data
function loadSettingsData() {
    fetchData('/api/admin/settings', (data) => {
        // Update settings UI
        console.log('Settings loaded:', data);
    });
}

// Knowledge Base data
function loadKnowledgeData() {
    // Knowledge base data would typically come from a dedicated API
    console.log('Knowledge base data would be loaded here');
}

// Logs data
function loadLogsData() {
    fetchData('/api/admin/logs', (data) => {
        if (data && data.success && data.logs) {
            // Update logs UI
            console.log('Logs loaded:', data);
        }
    });
}

// Search functionality
function performSearch(query) {
    if (!query) return;
    
    // Implementation would depend on what's being searched
    console.log('Search performed for:', query);
}

// Initialize charts
function initializeCharts() {
    // This would be implemented with a charting library
    console.log('Charts would be initialized here');
}

// Helper function for API calls
function fetchData(url, callback) {
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (callback) callback(data);
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
} 