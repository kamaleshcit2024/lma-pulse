import React, { useState, useEffect } from 'react';
import { TrendingUp, Activity } from 'lucide-react';
import apiService from '../services/api';
import './Forecaster.css';

const Forecaster = () => {
    const [forecast, setForecast] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadForecast();
    }, []);

    const loadForecast = async () => {
        try {
            const response = await apiService.getForecast(6);
            setForecast(response.data);
            setLoading(false);
        } catch (error) {
            console.error('Forecast load error:', error);
            setLoading(false);
        }
    };

    if (loading) return <div className="forecaster-loading">Loading AI Forecasts...</div>;

    return (
        <div className="forecaster-container">
            <div className="forecaster-header">
                <h2><Activity size={24} /> AI Financial Forecast (PerforatedAI)</h2>
                <p>Projected EBITDA metrics using adaptive forecasting</p>
            </div>

            <div className="forecast-grid">
                {forecast.map((item, index) => (
                    <div key={index} className="forecast-card">
                        <div className="forecast-month">Month +{item.month_offset}</div>
                        <div className="forecast-value">
                            ${(item.predicted_ebitda / 1_000_000).toFixed(1)}M
                        </div>
                        <div className="forecast-trend">
                            <TrendingUp size={16} /> Projected
                        </div>
                    </div>
                ))}
            </div>

            <div className="forecaster-note">
                <p>Powered by PerforatedAI: Neural network with adaptive dendritic connections automatically optimized during training.</p>
            </div>
        </div>
    );
};

export default Forecaster;
