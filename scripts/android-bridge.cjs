const http = require('http');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

const PORT = 8080;

function runAdb(cmd) {
  return new Promise((resolve) => {
    exec(`adb connect 127.0.0.1:5555 >/dev/null 2>&1; adb -s 127.0.0.1:5555 ${cmd}`, { timeout: 4500 }, (err, stdout, stderr) => {
      if (err) {
        if (String(stderr).includes('offline')) {
          exec('adb disconnect 127.0.0.1:5555 >/dev/null 2>&1; sleep 1; adb connect 127.0.0.1:5555 >/dev/null 2>&1');
        }
        resolve({ success: false, error: err.message, stderr: String(stderr) });
      } else {
        resolve({ success: true, stdout: (stdout || '').trim() });
      }
    });
  });
}

const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://${req.headers.host}`);

  // API: Wake & Unlock Phone
  if (url.pathname === '/api/wake') {
    await runAdb('shell input keyevent 82'); // Menu / Unlock
    await runAdb('shell input keyevent 3');  // Home
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true, message: 'Screen unlocked and Home screen requested' }));
    return;
  }

  // API: Reconnect ADB
  if (url.pathname === '/api/reconnect') {
    exec('adb disconnect 127.0.0.1:5555; sleep 1; adb connect 127.0.0.1:5555', () => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true }));
    });
    return;
  }

  // API: Health / Status
  if (url.pathname === '/api/status') {
    const boot = await runAdb('shell getprop sys.boot_completed');
    const isReady = boot.stdout === '1';
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      online: isReady,
      bootCompleted: isReady,
      timestamp: new Date().toISOString()
    }));
    return;
  }

  // API: Keyevent
  if (url.pathname === '/api/key') {
    const key = url.searchParams.get('k') || '4';
    const keyMap = {
      'back': '4',
      'home': '3',
      'appswitch': '187',
      'power': '26',
      'volup': '24',
      'voldown': '25',
      'enter': '66',
      'tab': '61',
      'backspace': '67',
      'menu': '82',
      'esc': '111'
    };
    const keycode = keyMap[key.toLowerCase()] || key;
    const result = await runAdb(`shell input keyevent ${keycode}`);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(result));
    return;
  }

  // API: Type text
  if (url.pathname === '/api/type') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const data = body ? JSON.parse(body) : {};
        const text = data.text || url.searchParams.get('t') || '';
        
        if (text) {
          const escaped = text.replace(/[%]/g, '%25')
                              .replace(/[ ]/g, '%s')
                              .replace(/[&]/g, '\\&')
                              .replace(/[<]/g, '\\<')
                              .replace(/[>]/g, '\\>')
                              .replace(/[(]/g, '\\(')
                              .replace(/[)]/g, '\\)')
                              .replace(/[']/g, "\\'")
                              .replace(/[""]/g, '\\"')
                              .replace(/[;]/g, '\\;');
          
          await runAdb(`shell input text "${escaped}"`);
          if (data.enter) {
            await runAdb('shell input keyevent 66');
          }
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true }));
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: err.message }));
      }
    });
    return;
  }

  // Fallback
  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Android Bridge API running on 127.0.0.1:${PORT}`);
});
