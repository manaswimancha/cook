import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [cookies, setCookies] = useState(() => {
    const savedCookies = localStorage.getItem('cookies');
    return savedCookies ? parseInt(savedCookies, 10) : 0;
  });

  useEffect(() => {
    localStorage.setItem('cookies', cookies);
  }, [cookies]);

  const handleClick = () => {
    setCookies(cookies + 1);
  };

  const handleReset = () => {
    setCookies(0);
  };

  return (
    <div className="App">
      <h1>Cookie Clicker</h1>
      <p>Cookies: {cookies}</p>
      <button onClick={handleClick} style={{ fontSize: '20px', padding: '10px' }}>
        Click the Cookie
      </button>
      <br />
      <button onClick={handleReset} style={{ marginTop: '20px', fontSize: '16px' }}>
        Reset Cookies
      </button>
    </div>
  );
}

export default App;
