import React from 'react'
import { Link, NavLink } from 'react-router-dom'

export default function NavBar() {
  return (
    <header className="w-full bg-base-100 shadow-sm">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-indigo-600 flex items-center justify-center text-white font-bold">HS</div>
          <span className="text-xl font-semibold">HateSpeech Detector</span>
        </Link>

        <nav className="hidden md:flex items-center gap-4">
          <NavLink to="/" className={({isActive}) => `px-3 py-2 rounded ${isActive? 'bg-indigo-100 text-indigo-700' : 'text-gray-700 hover:bg-gray-100'}`} end>Home</NavLink>
          <NavLink to="/about" className={({isActive}) => `px-3 py-2 rounded ${isActive? 'bg-indigo-100 text-indigo-700' : 'text-gray-700 hover:bg-gray-100'}`}>About</NavLink>
          <NavLink to="/contact" className={({isActive}) => `px-3 py-2 rounded ${isActive? 'bg-indigo-100 text-indigo-700' : 'text-gray-700 hover:bg-gray-100'}`}>Contact</NavLink>
          <NavLink to="/login" className="px-3 py-2 rounded bg-indigo-600 text-white hover:bg-indigo-500">Login</NavLink>
        </nav>

        <div className="md:hidden">
          <button className="btn btn-ghost">Menu</button>
        </div>
      </div>
    </header>
  )
}
