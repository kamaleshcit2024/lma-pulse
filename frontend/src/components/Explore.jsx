import React, { useState, useEffect } from 'react';
import Forecaster from './Forecaster';
import apiService from '../services/api';
import { DollarSign, Activity, TrendingUp } from 'lucide-react';
import './Dashboard.css';

const Explore = () => {
    const [financials, setFinancials] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadFinancialData();
    }, []);

    const loadFinancialData = async () => {
        try {
            const response = await apiService.getFinancialHistory(12);
            setFinancials(response.data);
            setLoading(false);
        } catch (error) {
            console.error('Failed to load financial history', error);
            setLoading(false);
        }
    };

    // Calculate Latest Metrics
    const latest = financials.length > 0 ? financials[financials.length - 1] : null;

    if (loading) return <div className="loading">Loading Financial Data...</div>;

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <div>
                    <h1>Explore Data</h1>
                    <p className="subtitle">Advanced Financial Forecasting & History</p>
                </div>
            </header>

            {/* Key Metrics Row */}
            {latest && (
                <div className="stats-grid">
                    <div className="stat-card">
                        <div className="stat-header">
                            <span className="stat-title">Current EBITDA</span>
                            <Activity size={20} className="stat-icon" />
                        </div>
                        <div className="stat-value">
                            ${(latest.ebitda / 1_000_000).toFixed(2)}M
                        </div>
                        <div className="stat-change positive">
                            Latest Actual
                        </div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-header">
                            <span className="stat-title">Total Net Debt</span>
                            <DollarSign size={20} className="stat-icon" />
                        </div>
                        <div className="stat-value">
                            ${(latest.total_debt / 1_000_000).toFixed(2)}M
                        </div>
                        <div className="stat-change custom">
                            Leverage: {(latest.total_debt / latest.ebitda).toFixed(2)}x
                        </div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-header">
                            <span className="stat-title">Cash Flow</span>
                            <TrendingUp size={20} className="stat-icon" />
                        </div>
                        <div className="stat-value">
                            ${(latest.cash_flow / 1_000_000).toFixed(2)}M
                        </div>
                        <div className="stat-change positive">
                            Operating Cash
                        </div>
                    </div>
                </div>
            )}

            <div className="grid-container" style={{ gridTemplateColumns: '1fr', marginTop: '2rem' }}>
                <div className="card">
                    <Forecaster />
                </div>

                <div className="card">
                    <h3>Historical Financial Performance</h3>
                    <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid #333' }}>
                                    <th style={{ padding: '1rem', color: '#a0a0a0' }}>Date</th>
                                    <th style={{ padding: '1rem', color: '#a0a0a0' }}>Revenue</th>
                                    <th style={{ padding: '1rem', color: '#a0a0a0' }}>EBITDA</th>
                                    <th style={{ padding: '1rem', color: '#a0a0a0' }}>Interest Exp</th>
                                    <th style={{ padding: '1rem', color: '#a0a0a0' }}>Net Profit</th>
                                </tr>
                            </thead>
                            <tbody>
                                {financials.slice().reverse().map((metric, index) => (
                                    <tr key={index} style={{ borderBottom: '1px solid #222' }}>
                                        <td style={{ padding: '1rem' }}>{new Date(metric.date).toLocaleDateString()}</td>
                                        <td style={{ padding: '1rem' }}>${(metric.revenue / 1_000_000).toFixed(2)}M</td>
                                        <td style={{ padding: '1rem' }}>${(metric.ebitda / 1_000_000).toFixed(2)}M</td>
                                        <td style={{ padding: '1rem' }}>${(metric.interest_expense / 1_000_000).toFixed(2)}M</td>
                                        <td style={{ padding: '1rem' }}>${(metric.net_profit / 1_000_000).toFixed(2)}M</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Explore;
