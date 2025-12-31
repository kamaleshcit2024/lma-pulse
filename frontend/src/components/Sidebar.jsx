
import React from 'react';
import { Home, FileText, Compass, LogIn, LogOut, Sliders } from 'lucide-react';
import './Sidebar.css';

const Sidebar = ({ activeTab, setActiveTab, isLoggedIn, onLogout }) => {
    return (
        <div className="sidebar">
            <div className="sidebar-header">
                <h2>LMA PULSE</h2>
            </div>

            <div className="sidebar-menu">
                <button
                    className={`menu-item ${activeTab === 'dashboard' ? 'active' : ''}`}
                    onClick={() => setActiveTab('dashboard')}
                >
                    <Home size={20} />
                    <span>Home</span>
                </button>

                <button
                    className={`menu-item ${activeTab === 'explore' ? 'active' : ''}`}
                    onClick={() => setActiveTab('explore')}
                >
                    <Compass size={20} />
                    <span>Explore</span>
                </button>

                <button
                    className={`menu-item ${activeTab === 'simulator' ? 'active' : ''}`}
                    onClick={() => setActiveTab('simulator')}
                >
                    <Sliders size={20} />
                    <span>Simulator</span>
                </button>

                <button
                    className={`menu-item ${activeTab === 'legal' ? 'active' : ''}`}
                    onClick={() => setActiveTab('legal')}
                >
                    <FileText size={20} />
                    <span>Document</span>
                </button>
            </div>

            <div className="sidebar-footer">
                {isLoggedIn ? (
                    <button className="menu-item logout" onClick={onLogout}>
                        <LogOut size={20} />
                        <span>Logout</span>
                    </button>
                ) : (
                    <button
                        className={`menu-item ${activeTab === 'login' ? 'active' : ''}`}
                        onClick={() => setActiveTab('login')}
                    >
                        <LogIn size={20} />
                        <span>Login</span>
                    </button>
                )}
            </div>
        </div>
    );
};

export default Sidebar;
