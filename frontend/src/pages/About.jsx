import React from 'react';

const About = () => {
  return (
    <div className="max-w-4xl mx-auto py-8">
      {/* Header */}
      <div className="card bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg mb-8">
        <div className="card-body">
          <h1 className="card-title text-4xl">About Hate Speech Detector</h1>
          <p className="text-lg mt-2">A machine learning-powered platform for identifying and analyzing harmful content online.</p>
        </div>
      </div>

      {/* Project Overview */}
      <div className="card bg-base-100 shadow-md mb-6">
        <div className="card-body">
          <h2 className="card-title text-2xl mb-4">Project Overview</h2>
          <p className="mb-3">
            The Hate Speech Detection System is a full-stack web application designed to identify and classify hate speech in text content. Our mission is to create safer online spaces by providing real-time analysis and classification of potentially harmful content.
          </p>
          <p className="mb-3">
            Built with modern technologies including React, Flask, and scikit-learn, our platform offers seamless integration, responsive design, and accurate ML-based predictions. Whether for content moderation, research, or platform safety, this tool provides actionable insights.
          </p>
          <p>
            We believe in the power of technology to foster positive communication while respecting free speech. Our detector is trained to identify patterns of hate speech while minimizing false positives.
          </p>
        </div>
      </div>

      {/* Features */}
      <div className="card bg-base-100 shadow-md mb-6">
        <div className="card-body">
          <h2 className="card-title text-2xl mb-4">Key Features</h2>
          <ul className="list-disc list-inside space-y-2">
            <li><strong>Real-time Detection:</strong> Analyze text instantly for hate speech patterns</li>
            <li><strong>ML-Powered:</strong> Scikit-learn models trained on diverse datasets</li>
            <li><strong>RESTful API:</strong> Easy integration with other applications</li>
            <li><strong>Responsive Design:</strong> Works flawlessly on desktop, tablet, and mobile</li>
            <li><strong>User-Friendly:</strong> Clean interface with intuitive controls</li>
            <li><strong>Detailed Results:</strong> Clear classification output and confidence metrics</li>
          </ul>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="card bg-base-100 shadow-md mb-6">
        <div className="card-body">
          <h2 className="card-title text-2xl mb-4">Technology Stack</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 className="font-semibold mb-2 text-indigo-600">Frontend</h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>React 19</li>
                <li>Vite (Build Tool)</li>
                <li>Tailwind CSS 4</li>
                <li>DaisyUI 5</li>
                <li>React Router v7</li>
                <li>Axios</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2 text-indigo-600">Backend</h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>Flask 3.1.1</li>
                <li>Python 3.10+</li>
                <li>scikit-learn</li>
                <li>Joblib</li>
                <li>Flask-CORS</li>
                <li>Flask-JWT-Extended</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Team */}
      <div className="card bg-base-100 shadow-md">
        <div className="card-body">
          <h2 className="card-title text-2xl mb-4">Our Team</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="card bg-gradient-to-br from-blue-50 to-blue-100 shadow">
              <div className="card-body text-center">
                <h3 className="font-semibold text-lg text-blue-800">Chaitanya Raj</h3>
                <p className="text-sm text-gray-600">Backend & ML Integration</p>
                <p className="text-xs text-gray-500 mt-2">Develops the core Flask API and ML model pipeline</p>
              </div>
            </div>
            <div className="card bg-gradient-to-br from-purple-50 to-purple-100 shadow">
              <div className="card-body text-center">
                <h3 className="font-semibold text-lg text-purple-800">Himanshu Singh</h3>
                <p className="text-sm text-gray-600">Frontend Development</p>
                <p className="text-xs text-gray-500 mt-2">Builds responsive UI with React and Tailwind CSS</p>
              </div>
            </div>
            <div className="card bg-gradient-to-br from-green-50 to-green-100 shadow">
              <div className="card-body text-center">
                <h3 className="font-semibold text-lg text-green-800">Ashish Ranjan</h3>
                <p className="text-sm text-gray-600">Infrastructure & Deployment</p>
                <p className="text-xs text-gray-500 mt-2">Manages data pipelines and system deployment</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;