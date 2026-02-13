/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
                display: ['Outfit', 'sans-serif'],
            },
            colors: {
                brand: {
                    gold: '#C5A059',
                    goldDark: '#9A7B3E',
                    black: '#0D0D0D',
                    panel: '#1A1A1A',
                },
                // Keeping legacy vars just in case, but focusing on brand.*
                background: '#020617',
                surface: '#0f172a',
                primary: {
                    DEFAULT: '#6366f1',
                    hover: '#4f46e5',
                    glow: '#818cf8',
                },
            },
            backgroundImage: {
                'luxury-gradient': 'linear-gradient(135deg, #C5A059 0%, #E5C578 50%, #C5A059 100%)',
            }
        },
    },
    plugins: [],
}
