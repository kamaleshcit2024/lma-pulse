import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { AlertTriangle, TrendingUp, DollarSign, FileText } from 'lucide-react';
import apiService from '../services/api';
import Forecaster from './Forecaster';
import './Dashboard.css';

const Dashboard = () => {
    const [borrowerProfile, setBorrowerProfile] = useState(null);
    const [financialHistory, setFinancialHistory] = useState([]);
    const [latestMetrics, setLatestMetrics] = useState(null);
    const [covenantTests, setCovenantTests] = useState([]);
    const [alerts, setAlerts] = useState({ breaches: [], warnings: [] });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadDashboardData();

        // Real-time polling every 30 seconds
        const interval = setInterval(loadDashboardData, 30000);
        return () => clearInterval(interval);
    }, []);

    const loadDashboardData = async () => {
        try {
            const [profile, history, latest, tests, alertsData] = await Promise.all([
                apiService.getBorrowerProfile(),
                apiService.getFinancialHistory(12),
                apiService.getLatestFinancials(),
                apiService.testCovenants(),
                apiService.getActiveAlerts(),
            ]);

            setBorrowerProfile(profile.data);
            setFinancialHistory(history.data);
            setLatestMetrics(latest.data);
            setCovenantTests(tests.data);
            setAlerts(alertsData.data);
            setLoading(false);
        } catch (error) {
            console.error('Dashboard load error:', error);
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="dashboard-loading">
                <div className="spinner"></div>
                <p>Loading covenant data...</p>
            </div>
        );
    }

    // Prepare chart data
    const leverageChartData = financialHistory.map(metric => ({
        date: new Date(metric.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }),
        leverage: (metric.total_debt / metric.ebitda).toFixed(2),
        threshold: 4.0
    }));

    const currentLeverage = latestMetrics ?
        (latestMetrics.total_debt / latestMetrics.ebitda).toFixed(2) : 0;

    return (
        <div className="dashboard">
            {/* Header */}
            <div className="dashboard-header">
                <div className="header-title">
                    <h1>LMA PULSE</h1>
                    <p className="subtitle">Real-Time Covenant Monitoring</p>
                </div>
                <div className="header-borrower">
                    <h2>{borrowerProfile?.name}</h2>
                    <p className="facility-details">
                        ${(borrowerProfile?.facility_amount / 1_000_000).toFixed(0)}M Facility
                        <span className="separator">•</span>
                        Agent: {borrowerProfile?.agent_bank}
                    </p>
                </div>
            </div>

            {/* Alert Banner */}
            {alerts.breaches.length > 0 && (
                <div className="alert-banner breach">
                    <AlertTriangle size={24} />
                    <div>
                        <strong>COVENANT BREACH DETECTED</strong>
                        <p>{alerts.breaches.length} covenant(s) breached - Immediate action required</p>
                    </div>
                </div>
            )}

            {alerts.warnings.length > 0 && alerts.breaches.length === 0 && (
                <div className="alert-banner warning">
                    <AlertTriangle size={20} />
                    <div>
                        <strong>WARNING</strong>
                        <p>{alerts.warnings.length} covenant(s) approaching threshold</p>
                    </div>
                </div>
            )}

            {/* Key Metrics Cards */}
            <div className="metrics-grid">
                <div className="metric-card">
                    <div className="metric-icon leverage">
                        <TrendingUp size={24} />
                    </div>
                    <div className="metric-content">
                        <p className="metric-label">Leverage Ratio</p>
                        <h3 className={`metric-value ${currentLeverage > 4.0 ? 'breach' : ''}`}>
                            {currentLeverage}x
                        </h3>
                        <p className="metric-threshold">Threshold: 4.00x</p>
                    </div>
                </div>

                <div className="metric-card">
                    <div className="metric-icon ebitda">
                        <DollarSign size={24} />
                    </div>
                    <div className="metric-content">
                        <p className="metric-label">EBITDA (TTM)</p>
                        <h3 className="metric-value">
                            ${(latestMetrics?.ebitda / 1_000_000).toFixed(1)}M
                        </h3>
                        <p className="metric-threshold">Total Debt: ${(latestMetrics?.total_debt / 1_000_000).toFixed(0)}M</p>
                    </div>
                </div>

                <div className="metric-card">
                    <div className="metric-icon cashflow">
                        <FileText size={24} />
                    </div>
                    <div className="metric-content">
                        <p className="metric-label">Free Cash Flow</p>
                        <h3 className="metric-value">
                            ${(latestMetrics?.cash_flow / 1_000_000).toFixed(1)}M
                        </h3>
                        <p className="metric-threshold">Interest Coverage: {(latestMetrics?.ebitda / (latestMetrics?.interest_expense * 12)).toFixed(2)}x</p>
                    </div>
                </div>
            </div>

            {/* Forecasting Component */}
            <Forecaster />

            {/* Leverage Ratio Chart */}
            <div className="chart-container">
                <h3>Clause 21.1 - Leverage Ratio Trend</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={leverageChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" stroke="#64748b" />
                        <YAxis stroke="#64748b" />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
                            labelStyle={{ color: '#e2e8f0' }}
                        />
                        <Legend />
                        <ReferenceLine y={4.0} stroke="#ef4444" strokeDasharray="3 3" label="Covenant Threshold" />
                        <Line type="monotone" dataKey="leverage" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Covenant Status Table */}
            <div className="covenant-table-container">
                <h3>Covenant Compliance Status</h3>
                <table className="covenant-table">
                    <thead>
                        <tr>
                            <th>Covenant</th>
                            <th>Clause</th>
                            <th>Threshold</th>
                            <th>Actual</th>
                            <th>Status</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {covenantTests.map((test, index) => (
                            <tr key={index} className={test.status}>
                                <td>
                                    {test.covenant_id === 'cov_leverage' && 'Leverage Ratio'}
                                    {test.covenant_id === 'cov_interest_cover' && 'Interest Cover'}
                                    {test.covenant_id === 'cov_debt_service' && 'Debt Service Cover'}
                                </td>
                                <td className="clause-ref">
                                    {test.covenant_id === 'cov_leverage' && 'Clause 21.1'}
                                    {test.covenant_id === 'cov_interest_cover' && 'Clause 21.2'}
                                    {test.covenant_id === 'cov_debt_service' && 'Clause 21.3'}
                                </td>
                                <td>{test.threshold_value}x</td>
                                <td className={test.status === 'breach' ? 'breach-value' : ''}>
                                    {test.actual_value}x
                                    {test.breach_margin && ` (${test.breach_margin > 0 ? '+' : ''}${test.breach_margin})`}
                                </td>
                                <td>
                                    <span className={`status-badge ${test.status}`}>
                                        {test.status.toUpperCase()}
                                    </span>
                                </td>
                                <td>
                                    {test.status === 'breach' && (
                                        <button className="action-button" onClick={() => handleGenerateLetter(test)}>
                                            Generate Letter
                                        </button>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );

    async function handleGenerateLetter(breach) {
        // This will be handled by the LegalDraftGenerator component
        window.dispatchEvent(new CustomEvent('generate-legal-doc', { detail: breach }));
    }
};

export default Dashboard;
