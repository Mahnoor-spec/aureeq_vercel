import './style.css'
import { appStore } from './lib/store';
import { Layout } from './components/Layout';
import { AgentWidget } from './components/AgentWidget';

// Demo Mode: Show only the Widget
const DemoPage = () => AgentWidget();

// Initial Render
function render() {
  const app = document.querySelector('#app');
  app.innerHTML = Layout(DemoPage());

  // Initialize 3D Avatar after DOM is inserted
  setTimeout(() => {
    const canvas = document.getElementById('avatar-canvas');
    if (canvas) {
      import('./lib/avatar/AvatarRenderer').then(({ AvatarRenderer }) => {
        const renderer = new AvatarRenderer(canvas);
        // Assuming AvatarSample_P.vrm is in public folder
        renderer.loadVRM('/avatar.vrm');

        // Setup Chat & Voice Interaction
        import('./components/AgentWidget').then(({ setupAgentInteraction }) => {
          // Export functions for AgentWidget to use
          window.avatarFunctions = {
            speakFromUrl: (url) => renderer.speakFromUrl(url)
          };
          setupAgentInteraction(renderer);
        });
      });
    }
  }, 100);
}

// Initial Render
render();

// Subscribe to store changes
appStore.subscribe(render);

// Global navigation helper
window.navigate = (view) => {
  appStore.setState({ view });
};
