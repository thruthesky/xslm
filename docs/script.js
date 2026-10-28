// Adds a Copy button to every code block.
document.querySelectorAll('.code').forEach((figure) => {
  const code = figure.querySelector('code');
  if (!code || !navigator.clipboard) return;
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'copy';
  button.textContent = 'Copy';
  button.addEventListener('click', async () => {
    await navigator.clipboard.writeText(code.textContent);
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = 'Copy'; }, 1500);
  });
  figure.append(button);
});
