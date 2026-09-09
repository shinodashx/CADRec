const tablist = document.querySelector('[role="tablist"]');
const tabs = [...tablist.querySelectorAll('[role="tab"]')];

function selectTab(selected) {
  for (const tab of tabs) {
    const active = tab === selected;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
  }
}

tablist.hidden = false;
selectTab(tabs[0]);
for (const [index, tab] of tabs.entries()) {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', (event) => {
    let next;
    if (event.key === 'ArrowRight') next = tabs[(index + 1) % tabs.length];
    if (event.key === 'ArrowLeft') next = tabs[(index - 1 + tabs.length) % tabs.length];
    if (event.key === 'Home') next = tabs[0];
    if (event.key === 'End') next = tabs[tabs.length - 1];
    if (!next) return;
    event.preventDefault();
    selectTab(next);
    next.focus();
  });
}

const copyButton = document.getElementById('copy-citation');
const copyStatus = document.getElementById('copy-status');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  const citation = document.getElementById('bibtex');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyStatus.textContent = 'Citation copied to clipboard.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Press Ctrl+C (or ⌘C on Mac) to copy.';
  }
});
