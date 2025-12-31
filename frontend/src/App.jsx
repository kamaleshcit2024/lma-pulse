import React, { useState } from 'react';
import Dashboard from './components/Dashboard';
import LegalDraftGenerator from './components/LegalDraftGenerator';
import Sidebar from './components/Sidebar';
import Login from './components/Login';
import Explore from './components/Explore';
import Simulator from './components/Simulator';
import './App.css';

function App() {
    const [activeTab, setActiveTab] = useState('login'); // Default to login
    const [isLoggedIn, setIsLoggedIn] = useState(false);

    const handleLogin = (username) => {
        setIsLoggedIn(true);
        setActiveTab('dashboard');
    };

    const handleLogout = () => {
        setIsLoggedIn(false);
        setActiveTab('login');
    };

    return (
        <div className="app">
            <Sidebar
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                isLoggedIn={isLoggedIn}
                onLogout={handleLogout}
            />

            <main className="app-main">
                {activeTab === 'login' && <Login onLogin={handleLogin} />}

                {/* Protected Routes - only show if logged in, or redirect/show login */}
                {isLoggedIn && activeTab === 'dashboard' && <Dashboard />}
                {isLoggedIn && activeTab === 'explore' && <Explore />}
                {isLoggedIn && activeTab === 'simulator' && <Simulator />}
                {isLoggedIn && activeTab === 'legal' && <LegalDraftGenerator />}

                {/* Fallback for protected routes if not logged in */}
                {!isLoggedIn && activeTab !== 'login' && (
                    <div className="dashboard-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                        <div className="card">
                            <h2>Access Restricted</h2>
                            <p>Please login to access this section.</p>
                            <button
                                className="action-button primary"
                                onClick={() => setActiveTab('login')}
                                style={{ marginTop: '1rem' }}
                            >
                                Go to Login
                            </button>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}

export default App;
