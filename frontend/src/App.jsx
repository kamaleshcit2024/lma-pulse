import React, { useState } from 'react';
import Dashboard from './components/Dashboard';
import LegalDraftGenerator from './components/LegalDraftGenerator';
import './App.css';

function App() {
    const [activeTab, setActiveTab] = useState('dashboard');

    return (
        <div className="app">
            <nav className="app-nav">
                <div className="nav-logo">
                    <h1>LMA PULSE</h1>
                </div>
                <div className="nav-tabs">
                    <button
                        className={activeTab === 'dashboard' ? 'active' : ''}
                        onClick={() => setActiveTab('dashboard')}
                    >
                        Dashboard
                    </button>
                    <button
                        className={activeTab === 'legal' ? 'active' : ''}
                        onClick={() => setActiveTab('legal')}
                    >
                        Legal Generator
                    </button>
                </div>
            </nav>

            <main className="app-main">
                {activeTab === 'dashboard' && <Dashboard />}
                {activeTab === 'legal' && <LegalDraftGenerator />}
            </main>
        </div>
    );
}

export default App;
