// Get the tail effect element and the page container
const tailEffectElement = document.querySelector('.tail-effect');
const pageContainer = document.body;

// Set the tail effect delay and easing function
const tailEffectDelay = 0.2;
const easingFunction = 'cubic-bezier(0.4, 0, 0.2, 1)';

// Add event listeners to track cursor movement
pageContainer.addEventListener('mousemove', (event) => {
  const x = event.clientX;
  const y = event.clientY;

  // Update the tail effect element's position with a delay
  requestAnimationFrame(() => {
    tailEffectElement.style.transform = `translate(${x}px, ${y}px)`;
    tailEffectElement.style.transition = `transform ${tailEffectDelay}s ${easingFunction}`;
  });
});

// Add event listeners to track cursor movement outside of navigation links
pageContainer.addEventListener('mouseleave', () => {
  tailEffectElement.style.transform = 'translate(0, 0)';
});

// Create a smooth trailing motion using a delay and easing function
let lastMouseX = 0;
let lastMouseY = 0;
let mouseX = 0;
let mouseY = 0;

function updateTailEffect() {
  mouseX += (lastMouseX - mouseX) * 0.1;
  mouseY += (lastMouseY - mouseY) * 0.1;

  tailEffectElement.style.transform = `translate(${mouseX}px, ${mouseY}px)`;

  requestAnimationFrame(updateTailEffect);
}

updateTailEffect();

// Handle multiple tail effect elements (optional)
const tailEffectElements = document.querySelectorAll('.tail-effect');

tailEffectElements.forEach((element) => {
  element.addEventListener('mousemove', (event) => {
    const x = event.clientX;
    const y = event.clientY;

    requestAnimationFrame(() => {
      element.style.transform = `translate(${x}px, ${y}px)`;
      element.style.transition = `transform ${tailEffectDelay}s ${easingFunction}`;
    });
  });
});

// Create a more complex animation path (optional)
function createAnimationPath() {
  const animationPath = [];
  const numPoints = 10;

  for (let i = 0; i < numPoints; i++) {
    const x = Math.random() * window.innerWidth;
    const y = Math.random() * window.innerHeight;
    animationPath.push({ x, y });
  }

  return animationPath;
}

const animationPath = createAnimationPath();

let animationIndex = 0;

function updateAnimation() {
  const point = animationPath[animationIndex];
  tailEffectElement.style.transform = `translate(${point.x}px, ${point.y}px)`;

  animationIndex = (animationIndex + 1) % animationPath.length;

  requestAnimationFrame(updateAnimation);
}

// Update the last mouse position on mouse move
pageContainer.addEventListener('mousemove', (event) => {
  lastMouseX = event.clientX;
  lastMouseY = event.clientY;
});

// Start the animation
updateAnimation();