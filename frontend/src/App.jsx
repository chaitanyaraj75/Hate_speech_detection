import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Outlet } from 'react-router-dom'
import Login from './components/Login';
import HateSpeechDetector from './pages/Hatespeechdetector.jsx';
import NavBar from './components/NavBar';
import About from './pages/About.jsx';
import Contact from './pages/Contact.jsx';

function App() {

  return (
    <>
     <div className="min-h-screen bg-base-200">
      <Router>
        <NavBar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<HateSpeechDetector />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/dashboard" element={<h1 className="text-2xl">Dashboard Page</h1>} />
            <Route path="/profile" element={<h1 className="text-2xl">Profile Page</h1>} />
            <Route path="/login" element={<Login />} />
            <Route path="*" element={<h1 className="text-2xl">404 Not Found</h1>} />
          </Routes>
        </main>
      </Router>
     </div>
    </>
  )
} 

export default App
