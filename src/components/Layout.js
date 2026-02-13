export function Layout(children) {
  return `
    <div class="h-screen w-full bg-[#050505] flex items-center justify-center font-sans overflow-hidden relative">
       <!-- Subtle Background Glow -->
       <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-brand-gold/5 rounded-full blur-[100px]"></div>
       
       <!-- The Widget (Centered for Demo) -->
       ${children}
    </div>
  `;
}
