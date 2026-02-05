import React, { useState } from 'react';

const Contact = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // For now, just log the form data
    console.log('Form submitted:', formData);
    alert('Thank you for your message. We will get back to you soon!');
    setFormData({ name: '', email: '', subject: '', message: '' });
  };

  const teamMembers = [
    {
      name: 'Chaitanya Raj',
      role: 'Backend & ML Integration',
      email: 'rajchaitanya75@gmail.com',
      phone: '+91 9570171234',
      bio: 'Full-stack developer specializing in ML pipelines and Flask APIs',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      name: 'Himanshu Singh',
      role: 'Frontend Development',
      email: 'himanshu.singh@gmail.com',
      phone: '+91 9876543210',
      bio: 'React specialist building responsive and modern UIs',
      color: 'from-purple-500 to-pink-500'
    },
    {
      name: 'Ashish Ranjan',
      role: 'Data & DevOps Engineer',
      email: 'ashish.ranjan@gmail.com',
      phone: '+91 9570171234',
      bio: 'DevOps engineer managing deployment and data pipelines',
      color: 'from-green-500 to-teal-500'
    }
  ];

  return (
    <div className="max-w-6xl mx-auto py-8">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-2">Contact Us</h1>
        <p className="text-gray-600">Have questions? Reach out to our team members below</p>
      </div>

      {/* Team Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {teamMembers.map((member, index) => (
          <div key={index} className="card bg-base-100 shadow-lg hover:shadow-xl transition-shadow">
            <div className={`bg-gradient-to-r ${member.color} h-32 flex items-center justify-center`}>
              <div className="text-white text-center">
                <div className="w-16 h-16 rounded-full bg-white/20 mx-auto flex items-center justify-center text-2xl font-bold">
                  {member.name.charAt(0)}
                </div>
              </div>
            </div>
            <div className="card-body">
              <h2 className="card-title text-lg">{member.name}</h2>
              <p className="text-sm text-indigo-600 font-semibold">{member.role}</p>
              <p className="text-sm text-gray-600 mt-2">{member.bio}</p>
              
              <div className="divider my-3"></div>
              
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <span className="text-indigo-600 font-semibold">Email:</span>
                  <a href={`mailto:${member.email}`} className="text-blue-500 hover:underline break-all">
                    {member.email}
                  </a>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-indigo-600 font-semibold">Phone:</span>
                  <a href={`tel:${member.phone}`} className="text-blue-500 hover:underline">
                    {member.phone}
                  </a>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Contact Form */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="card bg-base-100 shadow-lg">
          <div className="card-body">
            <h2 className="card-title text-2xl">Send us a Message</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="label">
                  <span className="label-text font-semibold">Full Name</span>
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  placeholder="Your name"
                  className="input input-bordered w-full"
                  required
                />
              </div>
              
              <div>
                <label className="label">
                  <span className="label-text font-semibold">Email</span>
                </label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="your.email@example.com"
                  className="input input-bordered w-full"
                  required
                />
              </div>

              <div>
                <label className="label">
                  <span className="label-text font-semibold">Subject</span>
                </label>
                <input
                  type="text"
                  name="subject"
                  value={formData.subject}
                  onChange={handleChange}
                  placeholder="Message subject"
                  className="input input-bordered w-full"
                  required
                />
              </div>

              <div>
                <label className="label">
                  <span className="label-text font-semibold">Message</span>
                </label>
                <textarea
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  placeholder="Your message here..."
                  rows="5"
                  className="textarea textarea-bordered w-full resize-none"
                  required
                ></textarea>
              </div>

              <button type="submit" className="btn btn-primary w-full">
                Send Message
              </button>
            </form>
          </div>
        </div>

        {/* Info Section */}
        <div className="space-y-6">
          <div className="card bg-indigo-50 shadow">
            <div className="card-body">
              <h3 className="card-title text-lg flex items-center gap-2 text-indigo-700">
                <span className="text-2xl">📧</span> Email Support
              </h3>
              <p className="text-sm text-black">For quick inquiries, you can reach individual team members through their contact information above.</p>
              <p className="text-sm mt-2 text-black">General inquiries: <strong>support@hatespeechdetector.com</strong></p>
            </div>
          </div>

          <div className="card bg-blue-50 shadow">
            <div className="card-body">
              <h3 className="card-title text-lg flex items-center gap-2 text-blue-700">
                <span className="text-2xl">🕐</span> Response Time
              </h3>
              <p className="text-sm text-black">We typically respond to inquiries within 24 business hours. Thank you for your patience!</p>
            </div>
          </div>

          <div className="card bg-green-50 shadow">
            <div className="card-body">
              <h3 className="card-title text-lg flex items-center gap-2 text-green-700">
                <span className="text-2xl">🔒</span> Privacy
              </h3>
              <p className="text-sm text-black">Your information is safe with us. We follow strict data protection guidelines and will never share your details.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact;
