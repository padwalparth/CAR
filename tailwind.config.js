/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: '#F7F7F2',
        surface: '#FFFFFF',
        ink: {
          DEFAULT: '#111111',
          muted: '#555555',
          light: '#888888',
        },
        brutal: {
          yellow: '#FFD84D',
          blue: '#4D7CFE',
          green: '#53D769',
          red: '#FF5A5F',
          purple: '#A78BFA',
          gray: '#EBEBE5',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Space Grotesk', 'Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace']
      },
      boxShadow: {
        'brutal-sm': '3px 3px 0px #111111',
        'brutal': '5px 5px 0px #111111',
        'brutal-lg': '8px 8px 0px #111111',
        'brutal-yellow': '5px 5px 0px #FFD84D',
        'brutal-blue': '5px 5px 0px #4D7CFE',
        'brutal-red': '5px 5px 0px #FF5A5F',
        'brutal-none': '0px 0px 0px #111111',
      },
      borderWidth: {
        '2': '2px',
        '3': '3px',
        '4': '4px',
      },
      borderRadius: {
        'none': '0px',
        'sm': '4px',
        'DEFAULT': '6px',
        'md': '6px',
        'lg': '8px',
        'xl': '10px',
        '2xl': '12px',
      }
    },
  },
  plugins: [],
}
