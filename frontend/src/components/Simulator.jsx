
import React, { useState, useEffect } from 'react';
import { Sliders, RefreshCw, AlertTriangle, CheckCircle, Activity, DollarSign, TrendingUp } from 'lucide-react';
import apiService from '../services/api';
import './Dashboard.css';
import './Simulator.css';

const Simulator = () => {
    // State for Sliders
    const [revenueChange, setRevenueChange] = useState(0); // -30 to +30 %
    const [interestChange, setInterestChange] = useState(0); // 0 to 500 bps

    // State for Results
    const [simulation, setSimulation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Initial Load & Debounced Simulation
    useEffect(() => {
        const timer = setTimeout(() => {
            runSimulation();
        }, 500); // 500ms debounce
        return () => clearTimeout(timer);
    }, [revenueChange, interestChange]);

    const runSimulation = async () => {
        setLoading(true);
        try {
            const params = {
                revenue_change_pct: revenueChange,
                interest_rate_change_bps: interestChange
            };
            const response = await apiService.simulateFinancials(params);
            setSimulation(response.data);
            setLoading(false);
        } catch (err) {
            console.error(err);
            setError('Simulation failed');
            setLoading(false);
        }
    };

    const resetSimulation = () => {
        setRevenueChange(0);
        setInterestChange(0);
    };

    // Helper to render Covenant Status
    const renderCovenantStatus = (results) => {
        const breachCount = results.filter(r => r.status === 'breach').length;
        if (breachCount === 0) return <span className="status-badge safe"><CheckCircle size={16} /> Compliant</span>;
        return <span className="status-badge danger"><AlertTriangle size={16} /> {breachCount} Breaches</span>;
    };

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <div>
                    <h1>Stress Testing Simulator</h1>
                    <p className="subtitle">Simulate adverse economic scenarios</p>
                </div>
                <button className="action-button secondary" onClick={resetSimulation}>
                    <RefreshCw size={16} /> Reset
                </button>
            </header>

            <div className="grid-container" style={{ gridTemplateColumns: '1fr 2fr', gap: '2rem' }}>
                {/* Controls Panel */}
                <div className="card">
                    <h3><Sliders size={20} /> Scenario Controls</h3>
                    <div style={{ marginTop: '2rem' }}>
                        <div className="slider-group">
                            <label className="slider-label">
                                <span>Revenue Shock</span>
                                <span className={revenueChange < 0 ? 'text-danger' : 'text-success'}>{revenueChange > 0 ? '+' : ''}{revenueChange}%</span>
                            </label>
                            <input
                                type="range"
                                min="-30"
                                max="30"
                                value={revenueChange}
                                onChange={(e) => setRevenueChange(parseInt(e.target.value))}
                                className="slider"
                            />
                            <small style={{ color: '#666' }}>Simulate impact of sales decline</small>
                        </div>

                        <div className="slider-group" style={{ marginTop: '2rem' }}>
                            <label className="slider-label">
                                <span>Interest Rate Hike</span>
                                <span className="text-warning">+{interestChange} bps</span>
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="500"
                                step="25"
                                value={interestChange}
                                onChange={(e) => setInterestChange(parseInt(e.target.value))}
                                className="slider"
                            />
                            <small style={{ color: '#666' }}>Simulate LIBOR/SOFR increases (100bps = 1%)</small>
                        </div>
                    </div>
                </div>

                {/* Results Panel */}
                <div className="card">
                    <h3>Simulation Results</h3>
                    {simulation && (
                        <div style={{ marginTop: '1.5rem' }}>
                            {/* Comparison Grid */}
                            <div className="comparison-grid">
                                <div className="result-metric">
                                    <span className="label">EBITDA</span>
                                    <div className="values">
                                        <span className="old">${(simulation.base_metrics.ebitda / 1000000).toFixed(1)}M</span>
                                        <span className="arrow">→</span>
                                        <span className={`new ${simulation.stressed_metrics.ebitda < simulation.base_metrics.ebitda ? 'negative' : ''}`}>
                                            ${(simulation.stressed_metrics.ebitda / 1000000).toFixed(1)}M
                                        </span>
                                    </div>
                                </div>
                                <div className="result-metric">
                                    <span className="label">Interest Exp</span>
                                    <div className="values">
                                        <span className="old">${(simulation.base_metrics.interest_expense / 1000000).toFixed(1)}M</span>
                                        <span className="arrow">→</span>
                                        <span className="new negative">
                                            ${(simulation.stressed_metrics.interest_expense / 1000000).toFixed(1)}M
                                        </span>
                                    </div>
                                </div>
                                <div className="result-metric">
                                    <span className="label">Net Debt</span>
                                    <div className="values">
                                        <span className="old">${(simulation.base_metrics.total_debt / 1000000).toFixed(1)}M</span>
                                        <span className="arrow">→</span>
                                        <span className="new">
                                            ${(simulation.stressed_metrics.total_debt / 1000000).toFixed(1)}M
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Covenant Impact */}
                            <div style={{ marginTop: '2rem', borderTop: '1px solid #333', paddingTop: '1rem' }}>
                                <h4>Covenant Impact</h4>
                                <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem', alignItems: 'center' }}>
                                    <div>Status: {renderCovenantStatus(simulation.covenant_results)}</div>
                                </div>

                                <div className="covenant-list-compact" style={{ marginTop: '1rem' }}>
                                    {simulation.covenant_results.map(res => (
                                        <div key={res.test_id} className={`covenant-item ${res.status}`}>
                                            <span className="name">{res.covenant_id}</span>
                                            <span className="val">{res.actual_value.toFixed(2)}x</span>
                                            <span className="threshold">(Limit: {res.threshold_value}x)</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Simulator;
