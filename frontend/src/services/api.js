import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const apiService = {
    // Borrower endpoints
    getBorrowerProfile: (id) => api.get(`/borrower/profile${id ? `?borrower_id=${id}` : ''}`),
    searchBorrowers: (query) => api.get(`/borrowers/search?q=${query}`),

    // Financial data endpoints
    getLatestFinancials: (id) => api.get(`/financial/latest${id ? `?borrower_id=${id}` : ''}`),
    getFinancialHistory: (months = 12, id) => api.get(`/financial/history?months=${months}${id ? `&borrower_id=${id}` : ''}`),

    // Covenant endpoints
    getCovenants: () => api.get('/covenants/list'),
    testCovenants: (id) => api.post(`/covenants/test${id ? `?borrower_id=${id}` : ''}`),

    // Legal document generation
    generateReservationOfRights: (breach, id) =>
        api.post(`/legal/generate-reservation${id ? `?borrower_id=${id}` : ''}`, breach),
    generateWaiverTemplate: (breach, id) =>
        api.post(`/legal/generate-waiver-template${id ? `?borrower_id=${id}` : ''}`, breach),

    // Alerts
    getActiveAlerts: (id) => api.get(`/alerts/active${id ? `?borrower_id=${id}` : ''}`),

    // Forecasting
    getForecast: (months = 3) => api.get(`/financial/forecast?months=${months}`),

    // Simulation
    simulateFinancials: (params) => api.post('/financial/simulate', params),
};

export default apiService;
