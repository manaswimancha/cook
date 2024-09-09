import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  // Initialize the cookie count from localStorage or default to 0
  const [cookies, setCookies] = useState(() => {
    const savedCookies = localStorage.getItem('cookies');
    return savedCookies ? parseInt(savedCookies, 10) : 0;
  });

  // Save the cookie count in localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('cookies', cookies);
  }, [cookies]);

  // Increment the cookie count when the button is clicked
  const handleClick = () => {
    setCookies(cookies + 1);
  };

  // Reset the cookie count
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
