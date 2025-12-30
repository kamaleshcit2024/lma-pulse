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
    getBorrowerProfile: () => api.get('/borrower/profile'),

    // Financial data endpoints
    getLatestFinancials: () => api.get('/financial/latest'),
    getFinancialHistory: (months = 12) => api.get(`/financial/history?months=${months}`),

    // Covenant endpoints
    getCovenants: () => api.get('/covenants/list'),
    testCovenants: () => api.post('/covenants/test'),

    // Legal document generation
    generateReservationOfRights: (breach) =>
        api.post('/legal/generate-reservation', breach),
    generateWaiverTemplate: (breach) =>
        api.post('/legal/generate-waiver-template', breach),

    // Alerts
    getActiveAlerts: () => api.get('/alerts/active'),

    // Forecasting
    getForecast: (months = 3) => api.get(`/financial/forecast?months=${months}`),
};

export default apiService;
