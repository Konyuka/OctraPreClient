#!/usr/bin/env python3
import json, base64, hashlib, time, sys, re, os, shutil, asyncio, aiohttp, threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import nacl.signing
import secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hmac
import ssl
import signal
import logging
import traceback

c = {'r': '\033[0m', 'b': '\033[34m', 'c': '\033[36m', 'g': '\033[32m', 'y': '\033[33m', 'R': '\033[31m', 'B': '\033[1m', 'bg': '\033[44m', 'bgr': '\033[41m', 'bgg': '\033[42m', 'w': '\033[37m'}

priv, addr, rpc = None, None, None
sk, pub = None, None
b58 = re.compile(r"^oct[1-9A-HJ-NP-Za-km-z]{44}$")
μ = 1_000_000
h = []
cb, cn, lu, lh = None, None, 0, 0
session = None
executor = ThreadPoolExecutor(max_workers=1)
stop_flag = threading.Event()
spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
spinner_idx = 0

# Auto mode variables
auto_mode = False
auto_stop_flag = threading.Event()
auto_addresses = [
    'octFmfT2tdqoXx1C9xJ83HM78Nc4hZbkahfcQrEwXBogqpb',
    'oct9sqLw4w4mrxvuSk92YikPTiE4gzRaKJvBU5owRAChNPn',
    'oct8JSKwc66sSYfA484ortpZC8GKmmd5YcZQSctMRVvs3Z2',
    'oct3EABVwSfXNg9dF5wWjU9enftZXovmdonbydA1yKdhEL5',
    'octGwk1keKx7x2S6A5vRqxxUW6jDUHYQJ8ZFf25PkqNjcp7',
]
auto_stats = {
    'send_tx': {'count': 0, 'last': None, 'next_idx': 0, 'errors': 0},
    'multi_send': {'count': 0, 'last': None, 'errors': 0},
    'private_transfer': {'count': 0, 'last': None, 'next_idx': 0, 'errors': 0},
    'encrypt_balance': {'count': 0, 'last': None, 'errors': 0},
    'decrypt_balance': {'count': 0, 'last': None, 'errors': 0},
    'auto_claim': {'count': 0, 'last': None, 'errors': 0}
}
auto_tasks = []

# Nonce management for auto mode to prevent duplicate transactions
nonce_lock = asyncio.Lock()
last_used_nonce = None

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

def sz():
    return shutil.get_terminal_size((80, 25))

def at(x, y, t, cl=''):
    print(f"\033[{y};{x}H{c['bg']}{cl}{t}{c['bg']}", end='')

def inp(x, y):
    print(f"\033[{y};{x}H", end='', flush=True)
    return input()

async def ainp(x, y):
    print(f"\033[{y};{x}H", end='', flush=True)
    try:
        # Block indefinitely without timeout to prevent unwanted CLI shutdowns
        return await asyncio.get_event_loop().run_in_executor(executor, input)
    except Exception as e:
        log_error("ainp", e, "Error in async input")
        stop_flag.set()
        return ''

def wait():
    cr = sz()
    msg = "press enter to continue..."
    msg_len = len(msg)
    y_pos = cr[1] - 2
    x_pos = max(2, (cr[0] - msg_len) // 2)
    at(x_pos, y_pos, msg, c['y'])
    print(f"\033[{y_pos};{x_pos + msg_len}H", end='', flush=True)
    input()

async def awaitkey():
    cr = sz()
    msg = "press enter to continue..."
    msg_len = len(msg)
    y_pos = cr[1] - 2
    x_pos = max(2, (cr[0] - msg_len) // 2)
    at(x_pos, y_pos, msg, c['y'])
    print(f"\033[{y_pos};{x_pos + msg_len}H{c['bg']}", end='', flush=True)
    try:
        # Block indefinitely without timeout to prevent unwanted CLI shutdowns
        await asyncio.get_event_loop().run_in_executor(executor, input)
    except Exception as e:
        log_error("awaitkey", e, "Error waiting for user input")
        stop_flag.set()

def ld():
    global priv, addr, rpc, sk, pub
    try:
        wallet_path = os.path.expanduser("~/.octra/wallet.json")
        if not os.path.exists(wallet_path):
            wallet_path = "wallet.json"
        
        with open(wallet_path, 'r') as f:
            d = json.load(f)
        
        priv = d.get('priv')
        addr = d.get('addr')
        rpc = d.get('rpc', 'http://localhost:8080')
        
        if not priv or not addr:
            return False
            
        if not rpc.startswith('https://') and 'localhost' not in rpc:
            print(f"{c['R']}⚠️  WARNING: Using insecure HTTP connection!{c['r']}")
            time.sleep(2)
            
        sk = nacl.signing.SigningKey(base64.b64decode(priv))
        pub = base64.b64encode(sk.verify_key.encode()).decode()
        return True
    except:
        return False

def fill():
    cr = sz()
    print(f"{c['bg']}", end='')
    for _ in range(cr[1]):
        print(" " * cr[0])
    print("\033[H", end='')

def box(x, y, w, h, t=""):
    print(f"\033[{y};{x}H{c['bg']}{c['w']}┌{'─' * (w - 2)}┐{c['bg']}")
    if t:
        print(f"\033[{y};{x}H{c['bg']}{c['w']}┤ {c['B']}{t} {c['w']}├{c['bg']}")
    for i in range(1, h - 1):
        print(f"\033[{y + i};{x}H{c['bg']}{c['w']}│{' ' * (w - 2)}│{c['bg']}")
    print(f"\033[{y + h - 1};{x}H{c['bg']}{c['w']}└{'─' * (w - 2)}┘{c['bg']}")

def format_time_ago(dt):
    """Format time ago string"""
    if not dt:
        return "never"
    
    diff = datetime.now() - dt
    seconds = int(diff.total_seconds())
    
    if seconds < 60:
        return f"{seconds}s ago"
    elif seconds < 3600:
        return f"{seconds//60}m ago"
    else:
        return f"{seconds//3600}h ago"

def format_next_action(last_time, interval_seconds):
    """Format next action time"""
    if not last_time:
        return "starting..."
    
    next_time = last_time + timedelta(seconds=interval_seconds)
    diff = next_time - datetime.now()
    seconds = int(diff.total_seconds())
    
    if seconds <= 0:
        return "now"
    elif seconds < 60:
        return f"{seconds}s"
    else:
        return f"{seconds//60}m {seconds%60}s"

async def spin_animation(x, y, msg):
    global spinner_idx
    try:
        while True:
            at(x, y, f"{c['c']}{spinner_frames[spinner_idx]} {msg}", c['c'])
            spinner_idx = (spinner_idx + 1) % len(spinner_frames)
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        at(x, y, " " * (len(msg) + 3), "")

def derive_encryption_key(privkey_b64):
    privkey_bytes = base64.b64decode(privkey_b64)
    salt = b"octra_encrypted_balance_v2"
    return hashlib.sha256(salt + privkey_bytes).digest()[:32]

def encrypt_client_balance(balance, privkey_b64):
    key = derive_encryption_key(privkey_b64)
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    plaintext = str(balance).encode()
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return "v2|" + base64.b64encode(nonce + ciphertext).decode()

def decrypt_client_balance(encrypted_data, privkey_b64):
    if encrypted_data == "0" or not encrypted_data:
        return 0
    
    if not encrypted_data.startswith("v2|"):
        privkey_bytes = base64.b64decode(privkey_b64)
        salt = b"octra_encrypted_balance_v1"
        key = hashlib.sha256(salt + privkey_bytes).digest() + hashlib.sha256(privkey_bytes + salt).digest()
        key = key[:32]
        
        try:
            data = base64.b64decode(encrypted_data)
            if len(data) < 32:
                return 0
            
            nonce = data[:16]
            tag = data[16:32]
            encrypted = data[32:]
            
            expected_tag = hashlib.sha256(nonce + encrypted + key).digest()[:16]
            if not hmac.compare_digest(tag, expected_tag):
                return 0
            
            decrypted = bytearray()
            key_hash = hashlib.sha256(key + nonce).digest()
            for i, byte in enumerate(encrypted):
                decrypted.append(byte ^ key_hash[i % 32])
            
            return int(decrypted.decode())
        except:
            return 0
    
    try:
        b64_data = encrypted_data[3:]
        raw = base64.b64decode(b64_data)
        
        if len(raw) < 28:
            return 0
        
        nonce = raw[:12]
        ciphertext = raw[12:]
        
        key = derive_encryption_key(privkey_b64)
        aesgcm = AESGCM(key)
        
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return int(plaintext.decode())
    except:
        return 0

def derive_shared_secret_for_claim(my_privkey_b64, ephemeral_pubkey_b64):
    sk = nacl.signing.SigningKey(base64.b64decode(my_privkey_b64))
    my_pubkey_bytes = sk.verify_key.encode()
    eph_pub_bytes = base64.b64decode(ephemeral_pubkey_b64)
    
    if eph_pub_bytes < my_pubkey_bytes:
        smaller, larger = eph_pub_bytes, my_pubkey_bytes
    else:
        smaller, larger = my_pubkey_bytes, eph_pub_bytes
    
    combined = smaller + larger
    round1 = hashlib.sha256(combined).digest()
    round2 = hashlib.sha256(round1 + b"OCTRA_SYMMETRIC_V1").digest()
    return round2[:32]

def decrypt_private_amount(encrypted_data, shared_secret):
    if not encrypted_data or not encrypted_data.startswith("v2|"):
        return None
    
    try:
        raw = base64.b64decode(encrypted_data[3:])
        if len(raw) < 28:
            return None
        
        nonce = raw[:12]
        ciphertext = raw[12:]
        
        aesgcm = AESGCM(shared_secret)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return int(plaintext.decode())
    except:
        return None

async def req(m, p, d=None, t=10):
    global session
    if not session:
        ssl_context = ssl.create_default_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context, force_close=True)
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=t),
            connector=connector,
            json_serialize=json.dumps
        )
    try:
        url = f"{rpc}{p}"
        
        kwargs = {}
        if m == 'POST' and d:
            kwargs['json'] = d
        
        log_info(f"Making {m} request to {url}")
        async with getattr(session, m.lower())(url, **kwargs) as resp:
            try:
                # Try to read as UTF-8 first
                text = await resp.text(encoding='utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try with error handling
                try:
                    text = await resp.text(encoding='utf-8', errors='replace')
                    log_warning(f"Request {m} {url} had encoding issues, using replacement characters")
                except:
                    # As a last resort, read as bytes and decode with error handling
                    raw_bytes = await resp.read()
                    text = raw_bytes.decode('utf-8', errors='replace')
                    log_warning(f"Request {m} {url} had severe encoding issues, forced decode")
            
            try:
                j = json.loads(text) if text.strip() else None
            except json.JSONDecodeError as e:
                log_warning(f"Request {m} {url} returned invalid JSON: {str(e)}")
                j = None
            except Exception as e:
                log_warning(f"Request {m} {url} JSON parsing error: {str(e)}")
                j = None
            
            log_info(f"Request {m} {url} completed with status {resp.status}")
            return resp.status, text, j
    except asyncio.TimeoutError:
        log_warning(f"Request {m} {url} timed out after {t}s")
        return 0, "timeout", None
    except Exception as e:
        log_error("req", e, f"Request {m} {url} failed")
        return 0, str(e), None

async def req_private(path, method='GET', data=None):
    global session
    if not session:
        ssl_context = ssl.create_default_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context, force_close=True)
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=connector,
            json_serialize=json.dumps
        )
    
    headers = {"X-Private-Key": priv}
    try:
        url = f"{rpc}{path}"
        
        kwargs = {'headers': headers}
        if method == 'POST' and data:
            kwargs['json'] = data
            
        log_info(f"Making private {method} request to {url}")
        async with getattr(session, method.lower())(url, **kwargs) as resp:
            try:
                # Try to read as UTF-8 first
                text = await resp.text(encoding='utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try with error handling
                try:
                    text = await resp.text(encoding='utf-8', errors='replace')
                    log_warning(f"Private request {method} {url} had encoding issues")
                except:
                    # As a last resort, read as bytes and decode with error handling
                    raw_bytes = await resp.read()
                    text = raw_bytes.decode('utf-8', errors='replace')
                    log_warning(f"Private request {method} {url} had severe encoding issues")
            
            if resp.status == 200:
                try:
                    log_info(f"Private request {method} {url} completed successfully")
                    return True, json.loads(text) if text.strip() else {}
                except json.JSONDecodeError as e:
                    log_warning(f"Private request {method} {url} returned invalid JSON: {str(e)}")
                    return False, {"error": "Invalid JSON response"}
                except Exception as e:
                    log_warning(f"Private request {method} {url} JSON parsing error: {str(e)}")
                    return False, {"error": "JSON parsing error"}
            else:
                log_warning(f"Private request {method} {url} returned status {resp.status}")
                return False, {"error": f"HTTP {resp.status}"}
                
    except Exception as e:
        log_error("req_private", e, f"Private request {method} {url} failed")
        return False, {"error": str(e)}

async def st():
    global cb, cn, lu
    now = time.time()
    if cb is not None and (now - lu) < 30:
        return cn, cb
    
    results = await asyncio.gather(
        req('GET', f'/balance/{addr}'),
        req('GET', '/staging', 5),
        return_exceptions=True
    )
    
    s, t, j = results[0] if not isinstance(results[0], Exception) else (0, str(results[0]), None)
    s2, t2, j2 = results[1] if not isinstance(results[1], Exception) else (0, str(results[1]), None)
    
    # Log staging request issues
    if isinstance(results[1], Exception):
        log_warning(f"Staging request failed with exception: {str(results[1])}")
    elif s2 != 200:
        log_warning(f"Staging request returned status {s2}: {t2}")
    
    if s == 200 and j:
        cn = int(j.get('nonce', 0))
        cb = float(j.get('balance', 0))
        lu = now
        if s2 == 200 and j2:
            our = [tx for tx in j2.get('staged_transactions', []) if tx.get('from') == addr]
            if our:
                cn = max(cn, max(int(tx.get('nonce', 0)) for tx in our))
    elif s == 404:
        cn, cb, lu = 0, 0.0, now
    elif s == 200 and t and not j:
        try:
            parts = t.strip().split()
            if len(parts) >= 2:
                cb = float(parts[0]) if parts[0].replace('.', '').isdigit() else 0.0
                cn = int(parts[1]) if parts[1].isdigit() else 0
                lu = now
            else:
                cn, cb = None, None
        except:
            cn, cb = None, None
    return cn, cb

async def get_encrypted_balance():
    ok, result = await req_private(f"/view_encrypted_balance/{addr}")
    
    if ok:
        try:
            return {
                "public": float(result.get("public_balance", "0").split()[0]),
                "public_raw": int(result.get("public_balance_raw", "0")),
                "encrypted": float(result.get("encrypted_balance", "0").split()[0]),
                "encrypted_raw": int(result.get("encrypted_balance_raw", "0")),
                "total": float(result.get("total_balance", "0").split()[0])
            }
        except Exception as e:
            log_warning(f"Error parsing encrypted balance data: {str(e)}, raw result: {result}")
            return None
    else:
        log_warning(f"Failed to get encrypted balance: {result}")
        return None

async def encrypt_balance(amount):
    enc_data = await get_encrypted_balance()
    if not enc_data:
        return False, {"error": "cannot get balance"}
    
    current_encrypted_raw = enc_data['encrypted_raw']
    new_encrypted_raw = current_encrypted_raw + int(amount * μ)
    
    encrypted_value = encrypt_client_balance(new_encrypted_raw, priv)
    
    # Add timestamp to prevent duplicate detection
    timestamp = str(int(time.time() * 1000000))  # microsecond timestamp
    
    data = {
        "address": addr,
        "amount": str(int(amount * μ)),
        "private_key": priv,
        "encrypted_data": encrypted_value,
        "timestamp": timestamp
    }
    
    s, t, j = await req('POST', '/encrypt_balance', data)
    if s == 200:
        return True, j
    else:
        return False, {"error": j.get("error", t) if j else t}

async def decrypt_balance(amount):
    enc_data = await get_encrypted_balance()
    if not enc_data:
        return False, {"error": "cannot get balance"}
    
    current_encrypted_raw = enc_data['encrypted_raw']
    if current_encrypted_raw < int(amount * μ):
        return False, {"error": "insufficient encrypted balance"}
    
    new_encrypted_raw = current_encrypted_raw - int(amount * μ)
    
    encrypted_value = encrypt_client_balance(new_encrypted_raw, priv)
    
    # Add timestamp to prevent duplicate detection
    timestamp = str(int(time.time() * 1000000))  # microsecond timestamp
    
    data = {
        "address": addr,
        "amount": str(int(amount * μ)),
        "private_key": priv,
        "encrypted_data": encrypted_value,
        "timestamp": timestamp
    }
    
    s, t, j = await req('POST', '/decrypt_balance', data)
    if s == 200:
        return True, j
    else:
        return False, {"error": j.get("error", t) if j else t}

async def get_address_info(address):
    s, t, j = await req('GET', f'/address/{address}')
    if s == 200:
        return j
    return None

async def get_public_key(address):
    s, t, j = await req('GET', f'/public_key/{address}')
    if s == 200:
        return j.get("public_key")
    return None

async def create_private_transfer(to_addr, amount):
    addr_info = await get_address_info(to_addr)
    if not addr_info or not addr_info.get("has_public_key"):
        return False, {"error": "Recipient has no public key"}
    
    to_public_key = await get_public_key(to_addr)
    if not to_public_key:
        return False, {"error": "Cannot get recipient public key"}
    
    # Add timestamp to prevent duplicate detection
    timestamp = str(int(time.time() * 1000000))  # microsecond timestamp
    
    data = {
        "from": addr,
        "to": to_addr,
        "amount": str(int(amount * μ)),
        "from_private_key": priv,
        "to_public_key": to_public_key,
        "timestamp": timestamp
    }
    
    s, t, j = await req('POST', '/private_transfer', data)
    if s == 200:
        return True, j
    else:
        return False, {"error": j.get("error", t) if j else t}

async def get_pending_transfers():
    ok, result = await req_private(f"/pending_private_transfers?address={addr}")
    
    if ok:
        transfers = result.get("pending_transfers", [])
        return transfers
    else:
        return []

async def claim_private_transfer(transfer_id):
    data = {
        "recipient_address": addr,
        "private_key": priv,
        "transfer_id": transfer_id
    }
    
    s, t, j = await req('POST', '/claim_private_transfer', data)
    if s == 200:
        return True, j
    else:
        return False, {"error": j.get("error", t) if j else t}

async def gh():
    global h, lh
    now = time.time()
    if now - lh < 60 and h:
        return
    s, t, j = await req('GET', f'/address/{addr}?limit=20')
    if s != 200 or (not j and not t):
        return
    
    if j and 'recent_transactions' in j:
        tx_hashes = [ref["hash"] for ref in j.get('recent_transactions', [])]
        tx_results = await asyncio.gather(*[req('GET', f'/tx/{hash}', 5) for hash in tx_hashes], return_exceptions=True)
        
        existing_hashes = {tx['hash'] for tx in h}
        nh = []
        
        for i, (ref, result) in enumerate(zip(j.get('recent_transactions', []), tx_results)):
            if isinstance(result, Exception):
                continue
            s2, _, j2 = result
            if s2 == 200 and j2 and 'parsed_tx' in j2:
                p = j2['parsed_tx']
                tx_hash = ref['hash']
                
                if tx_hash in existing_hashes:
                    continue
                
                ii = p.get('to') == addr
                ar = p.get('amount_raw', p.get('amount', '0'))
                a = float(ar) if '.' in str(ar) else int(ar) / μ
                msg = None
                if 'data' in j2:
                    try:
                        data = json.loads(j2['data'])
                        msg = data.get('message')
                    except:
                        pass
                nh.append({
                    'time': datetime.fromtimestamp(p.get('timestamp', 0)),
                    'hash': tx_hash,
                    'amt': a,
                    'to': p.get('to') if not ii else p.get('from'),
                    'type': 'in' if ii else 'out',
                    'ok': True,
                    'nonce': p.get('nonce', 0),
                    'epoch': ref.get('epoch', 0),
                    'msg': msg
                })
        
        oh = datetime.now() - timedelta(hours=1)
        h[:] = sorted(nh + [tx for tx in h if tx.get('time', datetime.now()) > oh], key=lambda x: x['time'], reverse=True)[:50]
        lh = now
    elif s == 404 or (s == 200 and t and 'no transactions' in t.lower()):
        h.clear()
        lh = now

def mk(to, a, n, msg=None):
    tx = {
        "from": addr,
        "to_": to,
        "amount": str(int(a * μ)),
        "nonce": int(n),
        "ou": "1" if a < 1000 else "3",
        "timestamp": time.time()
    }
    if msg:
        tx["message"] = msg
    bl = json.dumps({k: v for k, v in tx.items() if k != "message"}, separators=(",", ":"))
    sig = base64.b64encode(sk.sign(bl.encode()).signature).decode()
    tx.update(signature=sig, public_key=pub)
    return tx, hashlib.sha256(bl.encode()).hexdigest()

async def snd(tx):
    t0 = time.time()
    s, t, j = await req('POST', '/send-tx', tx)
    dt = time.time() - t0
    if s == 200:
        if j and j.get('status') == 'accepted':
            return True, j.get('tx_hash', ''), dt, j
        elif t.lower().startswith('ok'):
            return True, t.split()[-1], dt, None
    return False, json.dumps(j) if j else t, dt, j

async def expl(x, y, w, hb):
    box(x, y, w, hb, "wallet explorer")
    n, b = await st()
    await gh()
    at(x + 2, y + 2, "address:", c['c'])
    at(x + 11, y + 2, addr, c['w'])
    at(x + 2, y + 3, "balance:", c['c'])
    at(x + 11, y + 3, f"{b:.6f} oct" if b is not None else "---", c['B'] + c['g'] if b else c['w'])
    at(x + 2, y + 4, "nonce:  ", c['c'])
    at(x + 11, y + 4, str(n) if n is not None else "---", c['w'])
    at(x + 2, y + 5, "public: ", c['c'])
    at(x + 11, y + 5, pub[:40] + "...", c['w'])
    
    try:
        # Add timeout to prevent hanging
        enc_data = await asyncio.wait_for(get_encrypted_balance(), timeout=5.0)
        if enc_data:
            at(x + 2, y + 6, "encrypted:", c['c'])
            at(x + 13, y + 6, f"{enc_data['encrypted']:.6f} oct", c['B'] + c['y'])
            
            try:
                # Add timeout for pending transfers check
                pending = await asyncio.wait_for(get_pending_transfers(), timeout=5.0)
                if pending:
                    at(x + 2, y + 7, "claimable:", c['c'])
                    at(x + 13, y + 7, f"{len(pending)} transfers", c['B'] + c['g'])
            except asyncio.TimeoutError:
                log_warning("Timeout getting pending transfers")
    except asyncio.TimeoutError:
        log_warning("Timeout getting encrypted balance")
    except Exception as e:
        log_warning(f"Error getting encrypted balance: {str(e)}")
    
    try:
        # Add timeout for staging request with better error handling
        staging_result = await asyncio.wait_for(req('GET', '/staging', 2), timeout=3.0)
        if staging_result and staging_result[0] == 200 and staging_result[2]:
            j = staging_result[2]
            sc = len([tx for tx in j.get('staged_transactions', []) if tx.get('from') == addr])
        else:
            log_warning(f"Staging request failed: status={staging_result[0] if staging_result else 'None'}")
            sc = 0
    except asyncio.TimeoutError:
        log_warning("Timeout getting staging info")
        sc = 0
    except UnicodeDecodeError as e:
        log_warning(f"Staging request encoding error: {str(e)}")
        sc = 0
    except Exception as e:
        log_warning(f"Staging request error: {str(e)}")
        sc = 0
        
    at(x + 2, y + 8, "staging:", c['c'])
    at(x + 11, y + 8, f"{sc} pending" if sc else "none", c['y'] if sc else c['w'])
    at(x + 1, y + 9, "─" * (w - 2), c['w'])
    
    at(x + 2, y + 10, "recent transactions:", c['B'] + c['c'])
    if not h:
        at(x + 2, y + 12, "no transactions yet", c['y'])
    else:
        at(x + 2, y + 12, "time     type  amount      address", c['c'])
        at(x + 2, y + 13, "─" * (w - 4), c['w'])
        seen_hashes = set()
        display_count = 0
        sorted_h = sorted(h, key=lambda x: x['time'], reverse=True)
        for tx in sorted_h:
            if tx['hash'] in seen_hashes:
                continue
            seen_hashes.add(tx['hash'])
            if display_count >= min(len(h), hb - 17):
                break
            is_pending = not tx.get('epoch')
            time_color = c['y'] if is_pending else c['w']
            at(x + 2, y + 14 + display_count, tx['time'].strftime('%H:%M:%S'), time_color)
            at(x + 11, y + 14 + display_count, " in" if tx['type'] == 'in' else "out", c['g'] if tx['type'] == 'in' else c['R'])
            at(x + 16, y + 14 + display_count, f"{float(tx['amt']):>10.6f}", c['w'])
            at(x + 28, y + 14 + display_count, str(tx.get('to', '---')), c['y'])
            if tx.get('msg'):
                at(x + 77, y + 14 + display_count, "msg", c['c'])
            status_text = "pen" if is_pending else f"e{tx.get('epoch', 0)}"
            status_color = c['y'] + c['B'] if is_pending else c['c']
            at(x + w - 6, y + 14 + display_count, status_text, status_color)
            display_count += 1

def menu(x, y, w, h):
    box(x, y, w, h, "commands")
    at(x + 2, y + 2, "[1] send tx", c['w'])
    at(x + 2, y + 3, "[2] refresh", c['w'])
    at(x + 2, y + 4, "[3] multi send", c['w'])
    at(x + 2, y + 5, "[4] encrypt balance", c['w'])
    at(x + 2, y + 6, "[5] decrypt balance", c['w'])
    at(x + 2, y + 7, "[6] private transfer", c['w'])
    at(x + 2, y + 8, "[7] claim transfers", c['w'])
    at(x + 2, y + 9, "[8] export keys", c['w'])
    at(x + 2, y + 10, "[9] clear hist", c['w'])
    
    # Auto mode toggle
    auto_text = "stop auto" if auto_mode else "start auto"
    auto_color = c['R'] if auto_mode else c['g']
    at(x + 2, y + 11, f"[a] {auto_text}", auto_color)
    
    at(x + 2, y + 12, "[s] auto status", c['c'])
    at(x + 2, y + 13, "[0] exit", c['w'])
    at(x + 2, y + h - 2, "command: ", c['B'] + c['y'])

async def scr():
    cr = sz()
    cls()
    fill()
    t = f" octra client v0.1.0 (private) │ {datetime.now().strftime('%H:%M:%S')} "
    at((cr[0] - len(t)) // 2, 1, t, c['B'] + c['w'])
    
    sidebar_w = 28
    menu(2, 3, sidebar_w, 16)  # Increased height for new options
    
    info_y = 20  # Moved down by 1 to accommodate larger menu
    info_h = 11
    
    if auto_mode:
        # Show auto mode status instead of static info
        box(2, info_y, sidebar_w, info_h, "auto mode status")
        
        # Auto mode indicator
        at(4, info_y + 2, "AUTO MODE ACTIVE", c['bgg'] + c['w'])
        
        # Send TX status
        at(4, info_y + 3, f"send tx: {auto_stats['send_tx']['count']} sent", c['g'] if auto_stats['send_tx']['count'] > 0 else c['w'])
        at(4, info_y + 4, f"  next: {format_next_action(auto_stats['send_tx']['last'], 60)}", c['c'])
        
        # Multi send status
        at(4, info_y + 5, f"multi: {auto_stats['multi_send']['count']} sent", c['g'] if auto_stats['multi_send']['count'] > 0 else c['w'])
        at(4, info_y + 6, f"  next: {format_next_action(auto_stats['multi_send']['last'], 300)}", c['c'])
        
        # Private transfer status
        at(4, info_y + 7, f"private: {auto_stats['private_transfer']['count']} sent", c['g'] if auto_stats['private_transfer']['count'] > 0 else c['w'])
        
        # Encrypt/decrypt status
        encrypt_count = auto_stats['encrypt_balance']['count']
        decrypt_count = auto_stats['decrypt_balance']['count']
        at(4, info_y + 8, f"encrypt: {encrypt_count}/{decrypt_count}", c['y'])
        
        # Claim status
        at(4, info_y + 9, f"claimed: {auto_stats['auto_claim']['count']}", c['g'] if auto_stats['auto_claim']['count'] > 0 else c['w'])
        
    else:
        # Show static info when auto mode is off
        box(2, info_y, sidebar_w, info_h)
        at(4, info_y + 2, "testnet environment.", c['y'])
        at(4, info_y + 3, "actively updated.", c['y'])
        at(4, info_y + 4, "monitor changes!", c['y'])
        at(4, info_y + 5, "", c['y'])
        at(4, info_y + 6, "private transactions", c['g'])
        at(4, info_y + 7, "enabled", c['g'])
        at(4, info_y + 8, "", c['y'])
        at(4, info_y + 9, "tokens: no value", c['R'])
    
    explorer_x = sidebar_w + 4
    explorer_w = cr[0] - explorer_x - 2
    
    try:
        # Add timeout to prevent hanging on explorer section
        await asyncio.wait_for(expl(explorer_x, 3, explorer_w, cr[1] - 6), timeout=15.0)
    except asyncio.TimeoutError:
        log_warning("Explorer section timed out, continuing...")
        # Show error message in explorer area
        box(explorer_x, 3, explorer_w, cr[1] - 6, "wallet explorer (timeout)")
        at(explorer_x + 2, 5, "Network timeout - try refreshing", c['R'])
    except Exception as e:
        log_error("expl", e, "Error in explorer section")
        # Show error message in explorer area
        box(explorer_x, 3, explorer_w, cr[1] - 6, "wallet explorer (error)")
        at(explorer_x + 2, 5, f"Error: {str(e)[:50]}", c['R'])
    
    at(2, cr[1] - 1, " " * (cr[0] - 4), c['bg'])
    at(2, cr[1] - 1, "ready", c['bgg'] + c['w'])
    return await ainp(12, 17)  # Adjusted for new menu height

async def tx():
    cr = sz()
    cls()
    fill()
    w, hb = 85, 26
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    box(x, y, w, hb, "send transaction")
    at(x + 2, y + 2, "to address: (or [esc] to cancel)", c['y'])
    at(x + 2, y + 3, "─" * (w - 4), c['w'])
    to = await ainp(x + 2, y + 4)
    if not to or to.lower() == 'esc':
        return
    if not b58.match(to):
        at(x + 2, y + 14, "invalid address!", c['bgr'] + c['w'])
        at(x + 2, y + 15, "press enter to go back...", c['y'])
        await ainp(x + 2, y + 16)
        return
    at(x + 2, y + 5, f"to: {to}", c['g'])
    at(x + 2, y + 7, "amount: (or [esc] to cancel)", c['y'])
    at(x + 2, y + 8, "─" * (w - 4), c['w'])
    a = await ainp(x + 2, y + 9)
    if not a or a.lower() == 'esc':
        return
    if not re.match(r"^\d+(\.\d+)?$", a) or float(a) <= 0:
        at(x + 2, y + 14, "invalid amount!", c['bgr'] + c['w'])
        at(x + 2, y + 15, "press enter to go back...", c['y'])
        await ainp(x + 2, y + 16)
        return
    a = float(a)
    at(x + 2, y + 10, f"amount: {a:.6f} oct", c['g'])
    at(x + 2, y + 12, "message (optional, max 1024): (or enter to skip)", c['y'])
    at(x + 2, y + 13, "─" * (w - 4), c['w'])
    msg = await ainp(x + 2, y + 14)
    if not msg:
        msg = None
    elif len(msg) > 1024:
        msg = msg[:1024]
        at(x + 2, y + 15, "message truncated to 1024 chars", c['y'])
    
    global lu
    lu = 0
    n, b = await st()
    if n is None:
        at(x + 2, y + 17, "failed to get nonce!", c['bgr'] + c['w'])
        at(x + 2, y + 18, "press enter to go back...", c['y'])
        await ainp(x + 2, y + 19)
        return
    if not b or b < a:
        at(x + 2, y + 17, f"insufficient balance ({b:.6f} < {a})", c['bgr'] + c['w'])
        at(x + 2, y + 18, "press enter to go back...", c['y'])
        await ainp(x + 2, y + 19)
        return
    at(x + 2, y + 16, "─" * (w - 4), c['w'])
    at(x + 2, y + 17, f"send {a:.6f} oct", c['B'] + c['g'])
    at(x + 2, y + 18, f"to:  {to}", c['g'])
    if msg:
        at(x + 2, y + 19, f"msg: {msg[:50]}{'...' if len(msg) > 50 else ''}", c['c'])
    at(x + 2, y + 20, f"fee: {'0.001' if a < 1000 else '0.003'} oct (nonce: {n + 1})", c['y'])
    at(x + 2, y + 21, "[y]es / [n]o: ", c['B'] + c['y'])
    if (await ainp(x + 16, y + 21)).strip().lower() != 'y':
        return
    
    spin_task = asyncio.create_task(spin_animation(x + 2, y + 22, "sending transaction"))
    
    t, _ = mk(to, a, n + 1, msg)
    ok, hs, dt, r = await snd(t)
    
    spin_task.cancel()
    try:
        await spin_task
    except asyncio.CancelledError:
        pass
    
    if ok:
        for i in range(17, 25):
            at(x + 2, y + i, " " * (w - 4), c['bg'])
        at(x + 2, y + 20, f"✓ transaction accepted!", c['bgg'] + c['w'])
        at(x + 2, y + 21, f"hash: {hs[:64]}...", c['g'])
        at(x + 2, y + 22, f"      {hs[64:]}", c['g'])
        at(x + 2, y + 23, f"time: {dt:.2f}s", c['w'])
        if r and 'pool_info' in r:
            at(x + 2, y + 24, f"pool: {r['pool_info'].get('total_pool_size', 0)} txs pending", c['y'])
        h.append({
            'time': datetime.now(),
            'hash': hs,
            'amt': a,
            'to': to,
            'type': 'out',
            'ok': True,
            'msg': msg
        })
        lu = 0
    else:
        at(x + 2, y + 20, f"✗ transaction failed!", c['bgr'] + c['w'])
        at(x + 2, y + 21, f"error: {str(hs)[:w - 10]}", c['R'])
    await awaitkey()

async def multi():
    cr = sz()
    cls()
    fill()
    w, hb = 70, cr[1] - 4
    x = (cr[0] - w) // 2
    y = 2
    box(x, y, w, hb, "multi send")
    at(x + 2, y + 2, "enter recipients (address amount), empty line to finish:", c['y'])
    at(x + 2, y + 3, "type [esc] to cancel", c['c'])
    at(x + 2, y + 4, "─" * (w - 4), c['w'])
    rcp = []
    tot = 0
    ly = y + 5
    while ly < y + hb - 8:
        at(x + 2, ly, f"[{len(rcp) + 1}] ", c['c'])
        l = await ainp(x + 7, ly)
        if l.lower() == 'esc':
            return
        if not l:
            break
        p = l.split()
        if len(p) == 2 and b58.match(p[0]) and re.match(r"^\d+(\.\d+)?$", p[1]) and float(p[1]) > 0:
            a = float(p[1])
            rcp.append((p[0], a))
            tot += a
            at(x + 50, ly, f"+{a:.6f}", c['g'])
            ly += 1
        else:
            at(x + 50, ly, "invalid!", c['R'])
    if not rcp:
        return
    at(x + 2, y + hb - 7, "─" * (w - 4), c['w'])
    at(x + 2, y + hb - 6, f"total: {tot:.6f} oct to {len(rcp)} addresses", c['B'] + c['y'])
    global lu
    lu = 0
    n, b = await st()
    if n is None:
        at(x + 2, y + hb - 5, "failed to get nonce!", c['bgr'] + c['w'])
        at(x + 2, y + hb - 4, "press enter to go back...", c['y'])
        await ainp(x + 2, y + hb - 3)
        return
    if not b or b < tot:
        at(x + 2, y + hb - 5, f"insufficient balance! ({b:.6f} < {tot})", c['bgr'] + c['w'])
        at(x + 2, y + hb - 4, "press enter to go back...", c['y'])
        await ainp(x + 2, y + hb - 3)
        return
    at(x + 2, y + hb - 5, f"send all? [y/n] (starting nonce: {n + 1}): ", c['y'])
    if (await ainp(x + 48, y + hb - 5)).strip().lower() != 'y':
        return
    
    spin_task = asyncio.create_task(spin_animation(x + 2, y + hb - 3, "sending transactions"))
    
    batch_size = 5
    batches = [rcp[i:i+batch_size] for i in range(0, len(rcp), batch_size)]
    s_total, f_total = 0, 0
    
    for batch_idx, batch in enumerate(batches):
        tasks = []
        for i, (to, a) in enumerate(batch):
            idx = batch_idx * batch_size + i
            at(x + 2, y + hb - 2, f"[{idx + 1}/{len(rcp)}] preparing batch...", c['c'])
            t, _ = mk(to, a, n + 1 + idx)
            tasks.append(snd(t))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (result, (to, a)) in enumerate(zip(results, batch)):
            idx = batch_idx * batch_size + i
            if isinstance(result, Exception):
                f_total += 1
                at(x + 55, y + hb - 2, "✗ fail ", c['R'])
            else:
                ok, hs, _, _ = result
                if ok:
                    s_total += 1
                    at(x + 55, y + hb - 2, "✓ ok   ", c['g'])
                    h.append({
                        'time': datetime.now(),
                        'hash': hs,
                        'amt': a,
                        'to': to,
                        'type': 'out',
                        'ok': True
                    })
                else:
                    f_total += 1
                    at(x + 55, y + hb - 2, "✗ fail ", c['R'])
            at(x + 2, y + hb - 2, f"[{idx + 1}/{len(rcp)}] {a:.6f} to {to[:20]}...", c['c'])
            await asyncio.sleep(0.05)
    
    spin_task.cancel()
    try:
        await spin_task
    except asyncio.CancelledError:
        pass
    
    lu = 0
    at(x + 2, y + hb - 2, " " * 65, c['bg'])
    at(x + 2, y + hb - 2, f"completed: {s_total} success, {f_total} failed", c['bgg'] + c['w'] if f_total == 0 else c['bgr'] + c['w'])
    await awaitkey()

async def encrypt_balance_ui():
    cr = sz()
    cls()
    fill()
    w, hb = 70, 20
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    
    box(x, y, w, hb, "encrypt balance")
    
    _, pub_bal = await st()
    enc_data = await get_encrypted_balance()
    
    if not enc_data:
        at(x + 2, y + 10, "cannot get encrypted balance info", c['R'])
        await awaitkey()
        return
    
    at(x + 2, y + 2, "public balance:", c['c'])
    at(x + 20, y + 2, f"{pub_bal:.6f} oct", c['w'])
    
    at(x + 2, y + 3, "encrypted:", c['c'])
    at(x + 20, y + 3, f"{enc_data['encrypted']:.6f} oct", c['y'])
    
    at(x + 2, y + 4, "total:", c['c'])
    at(x + 20, y + 4, f"{enc_data['total']:.6f} oct", c['g'])
    
    at(x + 2, y + 6, "─" * (w - 4), c['w'])
    
    max_encrypt = enc_data['public_raw'] / μ - 1.0
    if max_encrypt <= 0:
        at(x + 2, y + 8, "insufficient public balance (need > 1 oct for fees)", c['R'])
        await awaitkey()
        return
    
    at(x + 2, y + 7, f"max encryptable: {max_encrypt:.6f} oct", c['y'])
    
    at(x + 2, y + 9, "amount to encrypt:", c['y'])
    amount = await ainp(x + 21, y + 9)
    
    if not amount or not re.match(r"^\d+(\.\d+)?$", amount) or float(amount) <= 0:
        return
    
    amount = float(amount)
    if amount > max_encrypt:
        at(x + 2, y + 11, f"amount too large (max: {max_encrypt:.6f})", c['R'])
        await awaitkey()
        return
    
    at(x + 2, y + 11, f"encrypt {amount:.6f} oct? [y/n]:", c['B'] + c['y'])
    if (await ainp(x + 30, y + 11)).strip().lower() != 'y':
        return
    
    spin_task = asyncio.create_task(spin_animation(x + 2, y + 14, "encrypting balance"))
    
    ok, result = await encrypt_balance(amount)
    
    spin_task.cancel()
    try:
        await spin_task
    except asyncio.CancelledError:
        pass
    
    if ok:
        at(x + 2, y + 14, "✓ encryption submitted!", c['bgg'] + c['w'])
        at(x + 2, y + 15, f"tx hash: {result.get('tx_hash', 'unknown')[:50]}...", c['g'])
        at(x + 2, y + 16, f"will process in next epoch", c['g'])
    else:
        at(x + 2, y + 14, f"✗ error: {result.get('error', 'unknown')}", c['bgr'] + c['w'])
    
    await awaitkey()

async def decrypt_balance_ui():
    cr = sz()
    cls()
    fill()
    w, hb = 70, 20
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    
    box(x, y, w, hb, "decrypt balance")
    
    _, pub_bal = await st()
    enc_data = await get_encrypted_balance()
    
    if not enc_data:
        at(x + 2, y + 10, "cannot get encrypted balance info", c['R'])
        await awaitkey()
        return
    
    at(x + 2, y + 2, "public balance:", c['c'])
    at(x + 20, y + 2, f"{pub_bal:.6f} oct", c['w'])
    
    at(x + 2, y + 3, "encrypted:", c['c'])
    at(x + 20, y + 3, f"{enc_data['encrypted']:.6f} oct", c['y'])
    
    at(x + 2, y + 4, "total:", c['c'])
    at(x + 20, y + 4, f"{enc_data['total']:.6f} oct", c['g'])
    
    at(x + 2, y + 6, "─" * (w - 4), c['w'])
    
    if enc_data['encrypted_raw'] == 0:
        at(x + 2, y + 8, "no encrypted balance to decrypt", c['R'])
        await awaitkey()
        return
    
    max_decrypt = enc_data['encrypted_raw'] / μ
    at(x + 2, y + 7, f"max decryptable: {max_decrypt:.6f} oct", c['y'])
    
    at(x + 2, y + 9, "amount to decrypt:", c['y'])
    amount = await ainp(x + 21, y + 9)
    
    if not amount or not re.match(r"^\d+(\.\d+)?$", amount) or float(amount) <= 0:
        return
    
    amount = float(amount)
    if amount > max_decrypt:
        at(x + 2, y + 11, f"amount too large (max: {max_decrypt:.6f})", c['R'])
        await awaitkey()
        return
    
    at(x + 2, y + 11, f"decrypt {amount:.6f} oct? [y/n]:", c['B'] + c['y'])
    if (await ainp(x + 30, y + 11)).strip().lower() != 'y':
        return
    
    spin_task = asyncio.create_task(spin_animation(x + 2, y + 14, "decrypting balance"))
    
    ok, result = await decrypt_balance(amount)
    
    spin_task.cancel()
    try:
        await spin_task
    except asyncio.CancelledError:
        pass
    
    if ok:
        at(x + 2, y + 14, "✓ decryption submitted!", c['bgg'] + c['w'])
        at(x + 2, y + 15, f"tx hash: {result.get('tx_hash', 'unknown')[:50]}...", c['g'])
        at(x + 2, y + 16, f"will process in next epoch", c['g'])
    else:
        at(x + 2, y + 14, f"✗ error: {result.get('error', 'unknown')}", c['bgr'] + c['w'])
    
    await awaitkey()

async def private_transfer_ui():
    cr = sz()
    cls()
    fill()
    w, hb = 80, 25
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    
    box(x, y, w, hb, "private transfer")
    
    enc_data = await get_encrypted_balance()
    if not enc_data or enc_data['encrypted_raw'] == 0:
        at(x + 2, y + 10, "no encrypted balance available", c['R'])
        at(x + 2, y + 11, "encrypt some balance first", c['y'])
        await awaitkey()
        return
    
    at(x + 2, y + 2, f"encrypted balance: {enc_data['encrypted']:.6f} oct", c['g'])
    at(x + 2, y + 3, "─" * (w - 4), c['w'])
    
    at(x + 2, y + 5, "recipient address:", c['y'])
    to_addr = await ainp(x + 2, y + 6)
    
    if not to_addr or not b58.match(to_addr):
        at(x + 2, y + 12, "invalid address", c['R'])
        await awaitkey()
        return
    
    if to_addr == addr:
        at(x + 2, y + 12, "cannot send to yourself", c['R'])
        await awaitkey()
        return
    
    spin_task = asyncio.create_task(spin_animation(x + 2, y + 8, "checking recipient"))
    
    addr_info = await get_address_info(to_addr)
    
    spin_task.cancel()
    try:
        await spin_task
    except asyncio.CancelledError:
        pass
    
    if not addr_info:
        at(x + 2, y + 12, "recipient address not found on blockchain", c['R'])
        await awaitkey()
        return
    
    if not addr_info.get('has_public_key'):
        at(x + 2, y + 12, "recipient has no public key", c['R'])
        at(x + 2, y + 13, "they need to make a transaction first", c['y'])
        await awaitkey()
        return
    
    at(x + 2, y + 8, f"recipient balance: {addr_info.get('balance', 'unknown')}", c['c'])
    
    at(x + 2, y + 10, "amount:", c['y'])
    amount = await ainp(x + 10, y + 10)
    
    if not amount or not re.match(r"^\d+(\.\d+)?$", amount) or float(amount) <= 0:
        return
    
    amount = float(amount)
    if amount > enc_data['encrypted'] :
        at(x + 2, y + 14, f"insufficient encrypted balance", c['R'])
        await awaitkey()
        return
    
    at(x + 2, y + 12, "─" * (w - 4), c['w'])
    at(x + 2, y + 13, f"send {amount:.6f} oct privately to", c['B'])
    at(x + 2, y + 14, to_addr, c['y'])
    at(x + 2, y + 16, "[y]es / [n]o:", c['B'] + c['y'])
    
    if (await ainp(x + 15, y + 16)).strip().lower() != 'y':
        return
    
    spin_task = asyncio.create_task(spin_animation(x + 2, y + 18, "creating private transfer"))
    
    ok, result = await create_private_transfer(to_addr, amount)
    
    spin_task.cancel()
    try:
        await spin_task
    except asyncio.CancelledError:
        pass
    
    if ok:
        at(x + 2, y + 18, "✓ private transfer submitted!", c['bgg'] + c['w'])
        at(x + 2, y + 19, f"tx hash: {result.get('tx_hash', 'unknown')[:50]}...", c['g'])
        at(x + 2, y + 20, f"recipient can claim in next epoch", c['g'])
        at(x + 2, y + 21, f"ephemeral key: {result.get('ephemeral_key', 'unknown')[:40]}...", c['c'])
    else:
        at(x + 2, y + 18, f"✗ error: {result.get('error', 'unknown')[:w-10]}", c['bgr'] + c['w'])
    
    await awaitkey()

async def claim_transfers_ui():
    cr = sz()
    cls()
    fill()
    w, hb = 85, cr[1] - 4
    x = (cr[0] - w) // 2
    y = 2
    
    box(x, y, w, hb, "claim private transfers")
    
    spin_task = asyncio.create_task(spin_animation(x + 2, y + 2, "loading pending transfers"))
    
    transfers = await get_pending_transfers()
    
    spin_task.cancel()
    try:
        await spin_task
    except asyncio.CancelledError:
        pass
    
    if not transfers:
        at(x + 2, y + 10, "no pending transfers", c['y'])
        await awaitkey()
        return
    
    at(x + 2, y + 2, f"found {len(transfers)} claimable transfers:", c['B'] + c['g'])
    at(x + 2, y + 4, "#   FROM                AMOUNT         EPOCH   ID", c['c'])
    at(x + 2, y + 5, "─" * (w - 4), c['w'])
    
    display_y = y + 6
    max_display = min(len(transfers), hb - 12)
    
    for i, t in enumerate(transfers[:max_display]):
        amount_str = "[encrypted]"
        amount_color = c['y']
        
        if t.get('encrypted_data') and t.get('ephemeral_key'):
            try:
                shared = derive_shared_secret_for_claim(priv, t['ephemeral_key'])
                amt = decrypt_private_amount(t['encrypted_data'], shared)
                if amt:
                    amount_str = f"{amt/μ:.6f} OCT"
                    amount_color = c['g']
            except:
                pass
        
        at(x + 2, display_y + i, f"[{i+1}]", c['c'])
        at(x + 8, display_y + i, t['sender'][:20] + "...", c['w'])
        at(x + 32, display_y + i, amount_str, amount_color)
        at(x + 48, display_y + i, f"ep{t.get('epoch_id', '?')}", c['c'])
        at(x + 58, display_y + i, f"#{t.get('id', '?')}", c['y'])
    
    if len(transfers) > max_display:
        at(x + 2, display_y + max_display + 1, f"... and {len(transfers) - max_display} more", c['y'])
    
    at(x + 2, y + hb - 6, "─" * (w - 4), c['w'])
    at(x + 2, y + hb - 5, "enter number to claim (0 to cancel):", c['y'])
    choice = await ainp(x + 40, y + hb - 5)
    
    if not choice or choice == '0':
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(transfers):
            transfer = transfers[idx]
            transfer_id = transfer['id']
            
            spin_task = asyncio.create_task(spin_animation(x + 2, y + hb - 3, f"claiming transfer #{transfer_id}"))
            
            ok, result = await claim_private_transfer(transfer_id)
            
            spin_task.cancel()
            try:
                await spin_task
            except asyncio.CancelledError:
                pass
            
            if ok:
                at(x + 2, y + hb - 3, f"✓ claimed {result.get('amount', 'unknown')}!", c['bgg'] + c['w'])
                at(x + 2, y + hb - 2, "your encrypted balance has been updated", c['g'])
            else:
                error_msg = result.get('error', 'unknown error')
                at(x + 2, y + hb - 3, f"✗ error: {error_msg[:w-10]}", c['bgr'] + c['w'])
        else:
            at(x + 2, y + hb - 3, "invalid selection", c['R'])
    except ValueError:
        at(x + 2, y + hb - 3, "invalid number", c['R'])
    except Exception:
        at(x + 2, y + hb - 3, f"error occurred", c['R'])
    
    await awaitkey()

async def exp():
    cr = sz()
    cls()
    fill()
    w, hb = 70, 15
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    box(x, y, w, hb, "export keys")
    
    at(x + 2, y + 2, "current wallet info:", c['c'])
    at(x + 2, y + 4, "address:", c['c'])
    at(x + 11, y + 4, addr[:32] + "...", c['w'])
    at(x + 2, y + 5, "balance:", c['c'])
    n, b = await st()
    at(x + 11, y + 5, f"{b:.6f} oct" if b is not None else "---", c['g'])
    
    at(x + 2, y + 7, "export options:", c['y'])
    at(x + 2, y + 8, "[1] show private key", c['w'])
    at(x + 2, y + 9, "[2] save full wallet to file", c['w'])
    at(x + 2, y + 10, "[3] copy address to clipboard", c['w'])
    at(x + 2, y + 11, "[0] cancel", c['w'])
    at(x + 2, y + 13, "choice: ", c['B'] + c['y'])
    
    choice = await ainp(x + 10, y + 13)
    choice = choice.strip()
    
    if choice == '1':
        at(x + 2, y + 7, " " * (w - 4), c['bg'])
        at(x + 2, y + 8, " " * (w - 4), c['bg'])
        at(x + 2, y + 9, " " * (w - 4), c['bg'])
        at(x + 2, y + 10, " " * (w - 4), c['bg'])
        at(x + 2, y + 11, " " * (w - 4), c['bg'])
        at(x + 2, y + 13, " " * (w - 4), c['bg'])
        
        at(x + 2, y + 7, "private key (keep secret!):", c['R'])
        at(x + 2, y + 8, priv[:32], c['R'])
        at(x + 2, y + 9, priv[32:], c['R'])
        at(x + 2, y + 11, "public key:", c['g'])
        at(x + 2, y + 12, pub[:44] + "...", c['g'])
        await awaitkey()
        
    elif choice == '2':
        fn = f"octra_wallet_{int(time.time())}.json"
        wallet_data = {
            'priv': priv,
            'addr': addr,
            'rpc': rpc
        }
        os.umask(0o077)
        with open(fn, 'w') as f:
            json.dump(wallet_data, f, indent=2)
        os.chmod(fn, 0o600)
        at(x + 2, y + 7, " " * (w - 4), c['bg'])
        at(x + 2, y + 8, " " * (w - 4), c['bg'])
        at(x + 2, y + 9, " " * (w - 4), c['bg'])
        at(x + 2, y + 10, " " * (w - 4), c['bg'])
        at(x + 2, y + 11, " " * (w - 4), c['bg'])
        at(x + 2, y + 13, " " * (w - 4), c['bg'])
        at(x + 2, y + 9, f"saved to {fn}", c['g'])
        at(x + 2, y + 11, "file contains private key - keep safe!", c['R'])
        await awaitkey()
        
    elif choice == '3':
        try:
            import pyperclip
            pyperclip.copy(addr)
            at(x + 2, y + 7, " " * (w - 4), c['bg'])
            at(x + 2, y + 9, "address copied to clipboard!", c['g'])
        except:
            at(x + 2, y + 7, " " * (w - 4), c['bg'])
            at(x + 2, y + 9, "clipboard not available", c['R'])
        at(x + 2, y + 11, " " * (w - 4), c['bg'])
        await awaitkey()

# Setup logging
def setup_logging():
    """Setup logging to file with rotation"""
    log_filename = f"octra_auto_mode_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create a custom logger
    logger = logging.getLogger('octra_auto')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler for critical errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

def log_error(func_name, error, additional_info=""):
    """Log error with full traceback"""
    error_msg = f"Error in {func_name}: {str(error)}"
    if additional_info:
        error_msg += f" | {additional_info}"
    
    logger.error(error_msg)
    logger.error("Full traceback:")
    logger.error(traceback.format_exc())

def log_info(message):
    """Log info message"""
    logger.info(message)

def log_warning(message):
    """Log warning message"""
    logger.warning(message)

def reset_auto_stats():
    """Reset auto mode statistics for a fresh start"""
    global auto_stats
    for key in auto_stats:
        auto_stats[key]['count'] = 0
        auto_stats[key]['errors'] = 0
        auto_stats[key]['last'] = None
        if 'next_idx' in auto_stats[key]:
            auto_stats[key]['next_idx'] = 0
    log_info("Auto mode statistics reset")

# Helper functions for auto mode transaction management
async def get_next_nonce():
    """Get the next available nonce with proper locking to prevent duplicates"""
    global last_used_nonce
    async with nonce_lock:
        current_nonce, _ = await st()
        if current_nonce is None:
            return None
        
        # For the first transaction, use current_nonce + 1
        # For subsequent transactions, increment from last_used_nonce
        if last_used_nonce is None:
            next_nonce = current_nonce + 1
        else:
            # Only increment if we're ahead of the server nonce
            next_nonce = max(current_nonce + 1, last_used_nonce + 1)
        
        last_used_nonce = next_nonce
        log_info(f"Allocated nonce {next_nonce} (current: {current_nonce}, last_used: {last_used_nonce})")
        return next_nonce

def is_duplicate_transaction_error(error_msg):
    """Check if the error is a duplicate transaction error"""
    if not error_msg:
        return False
    error_str = str(error_msg).lower()
    return ('duplicate' in error_str and 'transaction' in error_str) or 'duplicate transaction' in error_str

async def send_with_retry(target_addr, amount, tx_type, max_retries=3):
    """Send transaction with retry logic for duplicate transaction errors"""
    for attempt in range(max_retries):
        try:
            # Get fresh nonce for each attempt
            nonce = await get_next_nonce()
            if nonce is None:
                return False, "Failed to get nonce", None, None
            
            # Create and send transaction
            tx, tx_hash = mk(target_addr, amount, nonce, tx_type)
            ok, hs, dt, r = await snd(tx)
            
            if ok:
                log_info(f"Transaction successful on attempt {attempt + 1}: {hs}")
                return ok, hs, dt, r
            else:
                # Check if it's a duplicate transaction error
                if is_duplicate_transaction_error(hs):
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                        log_warning(f"Duplicate transaction detected on attempt {attempt + 1}, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        log_warning(f"Duplicate transaction error persisted after {max_retries} attempts, skipping...")
                        return False, f"Duplicate transaction after {max_retries} retries", dt, r
                else:
                    # Other error, don't retry
                    log_error("send_with_retry", f"Transaction failed with non-duplicate error: {hs}")
                    return ok, hs, dt, r
                    
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                log_warning(f"Transaction attempt {attempt + 1} failed with exception, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                log_error("send_with_retry", f"Transaction failed after {max_retries} attempts: {str(e)}")
                return False, str(e), None, None
    
    return False, "Max retries exceeded", None, None

# Auto mode functions
async def auto_send_tx():
    """Send 0.01 token to addresses in round robin every minute - limited to 2 executions"""
    log_info("Auto Send TX task started")
    max_executions = 2
    execution_count = 0
    
    try:
        while not auto_stop_flag.is_set() and execution_count < max_executions:
            try:
                target_addr = auto_addresses[auto_stats['send_tx']['next_idx']]
                log_info(f"Auto Send TX: Attempting to send 0.01 to {target_addr} (execution {execution_count + 1}/{max_executions})")
                
                # Check balance first
                _, b = await st()
                if b is None or b < 0.01:
                    auto_stats['send_tx']['errors'] += 1
                    log_warning(f"Auto Send TX: Insufficient balance ({b})")
                    await asyncio.sleep(60)
                    continue
                
                # Send transaction with retry logic
                ok, hs, dt, r = await send_with_retry(target_addr, 0.01, "auto_send", max_retries=3)
                
                if ok:
                    auto_stats['send_tx']['count'] += 1
                    auto_stats['send_tx']['last'] = datetime.now()
                    auto_stats['send_tx']['next_idx'] = (auto_stats['send_tx']['next_idx'] + 1) % len(auto_addresses)
                    log_info(f"Auto Send TX: Successfully sent 0.01 to {target_addr}, hash: {hs}")
                    
                    # Add to history
                    h.append({
                        'time': datetime.now(),
                        'hash': hs,
                        'amt': 0.01,
                        'to': target_addr,
                        'type': 'out',
                        'ok': True,
                        'msg': 'auto_send'
                    })
                    
                    # Increment execution count only on successful send
                    execution_count += 1
                    log_info(f"Auto Send TX: Completed execution {execution_count}/{max_executions}")
                    
                else:
                    auto_stats['send_tx']['errors'] += 1
                    # Only log as error if it's not a duplicate transaction that we're skipping
                    if not is_duplicate_transaction_error(hs):
                        log_error("auto_send_tx", Exception(f"Transaction failed: {hs}"), f"Failed to send to {target_addr}")
                    else:
                        log_info(f"Auto Send TX: Skipped duplicate transaction to {target_addr}")
                    
            except Exception as e:
                auto_stats['send_tx']['errors'] += 1
                log_error("auto_send_tx", e, f"Exception in send loop for {target_addr}")
            
            # Only wait if we haven't reached the limit
            if execution_count < max_executions:
                await asyncio.sleep(60)
        
        # Log completion
        if execution_count >= max_executions:
            log_info(f"Auto Send TX: Completed all {max_executions} executions, stopping")
        else:
            log_info("Auto Send TX: Stopped before completing all executions")
            
    except asyncio.CancelledError:
        log_info(f"Auto Send TX task cancelled after {execution_count}/{max_executions} executions")
    except Exception as e:
        log_error("auto_send_tx", e, "Fatal error in auto_send_tx")

async def auto_multi_send():
    """Multi send 0.01 token to all addresses every 5 minutes - limited to 1 execution"""
    log_info("Auto Multi Send task started")
    max_executions = 1
    execution_count = 0
    
    # Wait 10 seconds before starting to stagger with auto_send_tx
    await asyncio.sleep(10)
    log_info("Auto Multi Send: Starting after 10s delay")
    
    try:
        while not auto_stop_flag.is_set() and execution_count < max_executions:
            try:
                log_info(f"Auto Multi Send: Starting cycle - sending 0.01 to all addresses (execution {execution_count + 1}/{max_executions})")
                
                # Check balance first - we need at least 0.01 * number of addresses
                _, b = await st()
                required_balance = 0.01 * len(auto_addresses)
                if b is None or b < required_balance:
                    auto_stats['multi_send']['errors'] += 1
                    log_warning(f"Auto Multi Send: Insufficient balance ({b}) - need {required_balance} for all addresses")
                    await asyncio.sleep(300)  # 5 minutes
                    continue
                
                # Send to all addresses in this cycle
                all_success = True
                successful_sends = 0
                
                for i, target_addr in enumerate(auto_addresses):
                    try:
                        log_info(f"Auto Multi Send: Sending 0.01 to address {i + 1}/{len(auto_addresses)} ({target_addr})")
                        
                        # Small delay between transactions to avoid nonce conflicts
                        if i > 0:
                            await asyncio.sleep(2)
                        
                        # Send transaction with retry logic
                        ok, hs, dt, r = await send_with_retry(target_addr, 0.01, "auto_multi", max_retries=3)
                        
                        if ok:
                            successful_sends += 1
                            log_info(f"Auto Multi Send: Successfully sent 0.01 to {target_addr}, hash: {hs}")
                            
                            # Add to history
                            h.append({
                                'time': datetime.now(),
                                'hash': hs,
                                'amt': 0.01,
                                'to': target_addr,
                                'type': 'out',
                                'ok': True,
                                'msg': 'auto_multi'
                            })
                        else:
                            all_success = False
                            # Only log as error if it's not a duplicate transaction that we're skipping
                            if not is_duplicate_transaction_error(hs):
                                log_error("auto_multi_send", Exception(f"Transaction failed: {hs}"), f"Failed to send to {target_addr}")
                            else:
                                log_info(f"Auto Multi Send: Skipped duplicate transaction to {target_addr}")
                        
                    except Exception as e:
                        all_success = False
                        log_error("auto_multi_send", e, f"Exception sending to {target_addr}")
                
                # Update stats and increment execution count
                execution_count += 1
                
                if all_success:
                    auto_stats['multi_send']['count'] += 1
                    auto_stats['multi_send']['last'] = datetime.now()
                    log_info(f"Auto Multi Send: Cycle {execution_count}/{max_executions} completed successfully - sent to all {len(auto_addresses)} addresses")
                else:
                    auto_stats['multi_send']['errors'] += 1
                    log_error("auto_multi_send", Exception("Partial cycle failure"), f"Cycle {execution_count}/{max_executions} completed with errors - {successful_sends}/{len(auto_addresses)} successful")
                    
            except Exception as e:
                auto_stats['multi_send']['errors'] += 1
                log_error("auto_multi_send", e, "Exception in multi send cycle")
                execution_count += 1  # Still count as an execution even if it failed
            
            # Only wait if we haven't reached the limit
            if execution_count < max_executions:
                await asyncio.sleep(300)  # 5 minutes
        
        # Log completion
        if execution_count >= max_executions:
            log_info(f"Auto Multi Send: Completed all {max_executions} executions, stopping")
        else:
            log_info("Auto Multi Send: Stopped before completing all executions")
            
    except asyncio.CancelledError:
        log_info(f"Auto Multi Send task cancelled after {execution_count}/{max_executions} executions")
    except Exception as e:
        log_error("auto_multi_send", e, "Fatal error in auto_multi_send")

async def auto_private_transfer():
    """Private transfer 0.1 token to addresses in round robin every 5 minutes"""
    log_info("Auto private transfer started")
    consecutive_duplicates = 0
    try:
        while not auto_stop_flag.is_set():
            try:
                target_addr = auto_addresses[auto_stats['private_transfer']['next_idx']]
                log_info(f"Auto private transfer: attempting transfer to {target_addr}")
                
                # Check encrypted balance
                enc_data = await get_encrypted_balance()
                log_info(f"Auto private transfer: retrieved encrypted balance data: {enc_data}")
                if not enc_data or enc_data['encrypted'] < 0.1:
                    log_warning(f"Auto private transfer: insufficient encrypted balance ({enc_data['encrypted'] if enc_data else 'N/A'})")
                    # Don't count insufficient balance as an error, just skip this cycle
                    log_info("Auto private transfer: waiting 5 minutes before next attempt")
                    await asyncio.sleep(300)  # 5 minutes
                    continue
                
                # Small delay to avoid rapid succession
                await asyncio.sleep(1)
                
                # Create private transfer
                ok, result = await create_private_transfer(target_addr, 0.1)
                log_info(f"Auto private transfer: transfer result - ok: {ok}, result: {result}")
                
                if ok:
                    consecutive_duplicates = 0  # Reset counter on success
                    auto_stats['private_transfer']['count'] += 1
                    auto_stats['private_transfer']['last'] = datetime.now()
                    auto_stats['private_transfer']['next_idx'] = (auto_stats['private_transfer']['next_idx'] + 1) % len(auto_addresses)
                    log_info(f"Auto private transfer: successful transfer to {target_addr}, count: {auto_stats['private_transfer']['count']}")
                else:
                    # Check if it's a duplicate transaction error
                    if result and is_duplicate_transaction_error(result.get('error', '')):
                        consecutive_duplicates += 1
                        log_info(f"Auto private transfer: skipped duplicate transaction to {target_addr} (consecutive: {consecutive_duplicates})")
                        
                        # If we get too many consecutive duplicates, move to next address
                        if consecutive_duplicates >= 3:
                            log_warning(f"Auto private transfer: too many consecutive duplicates, moving to next address")
                            auto_stats['private_transfer']['next_idx'] = (auto_stats['private_transfer']['next_idx'] + 1) % len(auto_addresses)
                            consecutive_duplicates = 0
                    else:
                        consecutive_duplicates = 0  # Reset on non-duplicate error
                        auto_stats['private_transfer']['errors'] += 1
                        log_error("auto_private_transfer", Exception(f"Transfer failed: {result}"), f"Failed to transfer to {target_addr}")
                    
                # Always wait 5 minutes before next attempt
                log_info("Auto private transfer: waiting 5 minutes before next attempt")
                await asyncio.sleep(300)
                    
            except Exception as e:
                auto_stats['auto_claim']['errors'] += 1
                log_error("auto_private_transfer", e, f"Exception occurred: {str(e)}")
                # Wait 5 minutes before retrying
                await asyncio.sleep(300)
    except asyncio.CancelledError:
        log_info("Auto private transfer cancelled")
    except Exception as e:
        log_error(f"Auto private transfer: critical error: {str(e)}")
    finally:
        log_info("Auto private transfer stopped")

async def auto_encrypt_decrypt():
    """Encrypt 0.1 token every minute, decrypt the next minute"""
    log_info("Auto encrypt/decrypt started")
    try:
        encrypt_next = True
        while not auto_stop_flag.is_set():
            try:
                if encrypt_next:
                    # Encrypt balance
                    log_info("Auto encrypt/decrypt: attempting to encrypt 0.1 tokens")
                    enc_data = await get_encrypted_balance()
                    log_info(f"Auto encrypt: retrieved balance data: {enc_data}")
                    if enc_data and enc_data['public'] >= 1.1:  # Need extra for fees
                        ok, result = await encrypt_balance(0.1)
                        log_info(f"Auto encrypt: encryption result - ok: {ok}, result: {result}")
                        if ok:
                            auto_stats['encrypt_balance']['count'] += 1
                            auto_stats['encrypt_balance']['last'] = datetime.now()
                            log_info(f"Auto encrypt: successful encryption, count: {auto_stats['encrypt_balance']['count']}")
                        else:
                            # Check if it's a duplicate transaction error
                            if result and is_duplicate_transaction_error(result.get('error', '')):
                                log_info(f"Auto encrypt: skipped duplicate transaction")
                                # Don't count duplicates as errors
                            else:
                                auto_stats['encrypt_balance']['errors'] += 1
                                log_error("auto_encrypt_decrypt", Exception(f"Encryption failed: {result}"), "Failed to encrypt balance")
                    else:
                        # Don't count insufficient balance as an error
                        public_balance = enc_data['public'] if enc_data else 'N/A'
                        log_warning(f"Auto encrypt: insufficient public balance ({public_balance}), need at least 1.1")
                else:
                    # Decrypt balance
                    log_info("Auto encrypt/decrypt: attempting to decrypt 0.1 tokens")
                    enc_data = await get_encrypted_balance()
                    log_info(f"Auto decrypt: retrieved balance data: {enc_data}")
                    if enc_data and enc_data['encrypted'] >= 0.1:
                        ok, result = await decrypt_balance(0.1)
                        log_info(f"Auto decrypt: decryption result - ok: {ok}, result: {result}")
                        if ok:
                            auto_stats['decrypt_balance']['count'] += 1
                            auto_stats['decrypt_balance']['last'] = datetime.now()
                            log_info(f"Auto decrypt: successful decryption, count: {auto_stats['decrypt_balance']['count']}")
                        else:
                            # Check if it's a duplicate transaction error
                            if result and is_duplicate_transaction_error(result.get('error', '')):
                                log_info(f"Auto decrypt: skipped duplicate transaction")
                                # Don't count duplicates as errors
                            else:
                                auto_stats['decrypt_balance']['errors'] += 1
                                log_error("auto_encrypt_decrypt", Exception(f"Decryption failed: {result}"), "Failed to decrypt balance")
                    else:
                        # Don't count insufficient balance as an error
                        encrypted_balance = enc_data['encrypted'] if enc_data else 'N/A'
                        log_warning(f"Auto decrypt: insufficient encrypted balance ({encrypted_balance}), need at least 0.1")
                
                encrypt_next = not encrypt_next
                log_info("Auto encrypt/decrypt: waiting 60 seconds before next attempt")
                await asyncio.sleep(60)
                    
            except Exception as e:
                if encrypt_next:
                    auto_stats['encrypt_balance']['errors'] += 1
                    log_error("auto_encrypt_decrypt", e, f"Exception in encrypt: {str(e)}")
                else:
                    auto_stats['decrypt_balance']['errors'] += 1
                    log_error("auto_encrypt_decrypt", e, f"Exception in decrypt: {str(e)}")
                await asyncio.sleep(60)
    except asyncio.CancelledError:
        log_info("Auto encrypt/decrypt cancelled")
    except Exception as e:
        log_error(f"Auto encrypt/decrypt: critical error: {str(e)}")
    finally:
        log_info("Auto encrypt/decrypt stopped")

async def auto_claim():
    """Auto claim private transfers when available"""
    log_info("Auto claim started")
    try:
        while not auto_stop_flag.is_set():
            try:
                transfers = await get_pending_transfers()
                
                if transfers:
                    log_info(f"Auto claim: found {len(transfers)} pending transfers")
                else:
                    log_info("Auto claim: no pending transfers found")
                
                for transfer in transfers:
                    try:
                        transfer_id = transfer['id']
                        log_info(f"Auto claim: attempting to claim transfer {transfer_id}")
                        ok, result = await claim_private_transfer(transfer_id)
                        
                        if ok:
                            auto_stats['auto_claim']['count'] += 1
                            auto_stats['auto_claim']['last'] = datetime.now()
                            log_info(f"Auto claim: successfully claimed transfer {transfer_id}, count: {auto_stats['auto_claim']['count']}")
                        else:
                            auto_stats['auto_claim']['errors'] += 1
                            log_error(f"Auto claim: failed to claim transfer {transfer_id}, result: {result}")
                            
                        # Small delay between claims
                        await asyncio.sleep(1)
                    except Exception as e:
                        auto_stats['auto_claim']['errors'] += 1
                        log_error(f"Auto claim: exception while claiming transfer {transfer.get('id', 'unknown')}: {str(e)}")
                        
            except Exception as e:
                auto_stats['auto_claim']['errors'] += 1
                log_error(f"Auto claim: exception while getting pending transfers: {str(e)}")
            
            # Check for new transfers every 30 seconds
            await asyncio.sleep(30)
    except asyncio.CancelledError:
        log_info("Auto claim cancelled")
    except Exception as e:
        log_error(f"Auto claim: critical error: {str(e)}")
    finally:
        log_info("Auto claim stopped")

async def start_auto_mode():
    """Start all auto mode tasks"""
    global auto_tasks, auto_mode
    if auto_mode:
        log_info("Auto mode already running")
        return
    
    try:
        log_info("Starting auto mode...")
        print("Starting auto mode...")
        auto_mode = True
        auto_stop_flag.clear()
        
        # Reset nonce counter to avoid stale state
        global last_used_nonce
        last_used_nonce = None
        
        # Reset statistics for a fresh start
        reset_auto_stats()
        
        # Create individual tasks and store them separately
        auto_tasks = []
        task_names = ['send_tx', 'multi_send', 'private_transfer', 'encrypt_decrypt', 'claim']
        task_functions = [auto_send_tx, auto_multi_send, auto_private_transfer, auto_encrypt_decrypt, auto_claim]
        
        for name, func in zip(task_names, task_functions):
            try:
                log_info(f"Starting auto {name} task")
                task = asyncio.create_task(func())
                task.set_name(f"auto_{name}")
                auto_tasks.append(task)
                print(f"✓ Started auto {name}")
                log_info(f"Successfully started auto {name}")
            except Exception as e:
                error_msg = f"Failed to start auto {name}: {e}"
                print(f"✗ {error_msg}")
                log_error(f"start_auto_mode", e, f"Failed to start {name}")
        
        if auto_tasks:
            success_msg = f"Auto mode started with {len(auto_tasks)} active tasks"
            print(f"✓ {success_msg}")
            log_info(success_msg)
        else:
            auto_mode = False
            error_msg = "No auto tasks could be started"
            print(f"✗ {error_msg}")
            log_error("start_auto_mode", Exception(error_msg), "No tasks started")
            
    except Exception as e:
        auto_mode = False
        error_msg = f"Auto mode startup failed: {e}"
        print(f"✗ {error_msg}")
        log_error("start_auto_mode", e, "Startup failed")
        raise e

async def stop_auto_mode():
    """Stop auto mode"""
    global auto_tasks, auto_mode
    if not auto_mode:
        return
    
    try:
        print("Stopping auto mode...")
        log_info("Stopping auto mode...")
        auto_mode = False
        auto_stop_flag.set()
        
        if auto_tasks:
            # Cancel all auto tasks
            for task in auto_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to finish
            await asyncio.gather(*auto_tasks, return_exceptions=True)
            auto_tasks.clear()
        
        print("✓ Auto mode stopped")
        log_info("Auto mode stopped successfully")
        
    except Exception as e:
        print(f"✗ Error stopping auto mode: {e}")
        log_error("stop_auto_mode", e, "Error during shutdown")
        # Force reset even if there was an error
        auto_mode = False
        if auto_tasks:
            auto_tasks.clear()

async def auto_status():
    """Show detailed auto mode status"""
    cr = sz()
    cls()
    fill()
    w, hb = 85, cr[1] - 4
    x = (cr[0] - w) // 2
    y = 2
    
    box(x, y, w, hb, "auto mode status")
    
    if not auto_mode:
        at(x + 2, y + 10, "auto mode is not running", c['R'])
        at(x + 2, y + 12, "press [a] to start auto mode", c['y'])
        await awaitkey()
        return
    
    at(x + 2, y + 2, "AUTO MODE ACTIVE", c['bgg'] + c['w'])
    at(x + 2, y + 4, "TARGET ADDRESSES:", c['B'] + c['c'])
    
    for i, addr in enumerate(auto_addresses):
        at(x + 4, y + 5 + i, f"[{i+1}] {addr}", c['w'])
    
    at(x + 2, y + 11, "ACTIVITY STATUS:", c['B'] + c['c'])
    
    # Send TX status
    at(x + 4, y + 13, "Send TX (every 1 min):", c['c'])
    at(x + 26, y + 13, f"count: {auto_stats['send_tx']['count']}", c['g'])
    at(x + 40, y + 13, f"errors: {auto_stats['send_tx']['errors']}", c['R'] if auto_stats['send_tx']['errors'] > 0 else c['w'])
    at(x + 55, y + 13, f"last: {format_time_ago(auto_stats['send_tx']['last'])}", c['y'])
    
    # Multi send status
    at(x + 4, y + 14, "Multi Send (every 5 min):", c['c'])
    at(x + 26, y + 14, f"count: {auto_stats['multi_send']['count']}", c['g'])
    at(x + 40, y + 14, f"errors: {auto_stats['multi_send']['errors']}", c['R'] if auto_stats['multi_send']['errors'] > 0 else c['w'])
    at(x + 55, y + 14, f"last: {format_time_ago(auto_stats['multi_send']['last'])}", c['y'])
    
    # Private transfer status
    at(x + 4, y + 15, "Private Transfer (every 5 min):", c['c'])
    at(x + 26, y + 15, f"count: {auto_stats['private_transfer']['count']}", c['g'])
    at(x + 40, y + 15, f"errors: {auto_stats['private_transfer']['errors']}", c['R'] if auto_stats['private_transfer']['errors'] > 0 else c['w'])
    at(x + 55, y + 15, f"last: {format_time_ago(auto_stats['private_transfer']['last'])}", c['y'])
    
    # Encrypt/Decrypt status
    at(x + 4, y + 16, "Encrypt Balance (every 1 min):", c['c'])
    at(x + 26, y + 16, f"count: {auto_stats['encrypt_balance']['count']}", c['g'])
    at(x + 40, y + 16, f"errors: {auto_stats['encrypt_balance']['errors']}", c['R'] if auto_stats['encrypt_balance']['errors'] > 0 else c['w'])
    at(x + 55, y + 16, f"last: {format_time_ago(auto_stats['encrypt_balance']['last'])}", c['y'])
    
    at(x + 4, y + 17, "Decrypt Balance (every 1 min):", c['c'])
    at(x + 26, y + 17, f"count: {auto_stats['decrypt_balance']['count']}", c['g'])
    at(x + 40, y + 17, f"errors: {auto_stats['decrypt_balance']['errors']}", c['R'] if auto_stats['decrypt_balance']['errors'] > 0 else c['w'])
    at(x + 55, y + 17, f"last: {format_time_ago(auto_stats['decrypt_balance']['last'])}", c['y'])
    
    # Auto claim status
    at(x + 4, y + 18, "Auto Claim (continuous):", c['c'])
    at(x + 26, y + 18, f"count: {auto_stats['auto_claim']['count']}", c['g'])
    at(x + 40, y + 18, f"errors: {auto_stats['auto_claim']['errors']}", c['R'] if auto_stats['auto_claim']['errors'] > 0 else c['w'])
    at(x + 55, y + 18, f"last: {format_time_ago(auto_stats['auto_claim']['last'])}", c['y'])
    
    # Next actions
    at(x + 2, y + 20, "NEXT ACTIONS:", c['B'] + c['c'])
    at(x + 4, y + 21, f"Send TX: {format_next_action(auto_stats['send_tx']['last'], 60)}", c['y'])
    at(x + 30, y + 21, f"Multi Send: {format_next_action(auto_stats['multi_send']['last'], 300)}", c['y'])
    at(x + 60, y + 21, f"Private: {format_next_action(auto_stats['private_transfer']['last'], 300)}", c['y'])
    
    await awaitkey()

def signal_handler(sig, frame):
    stop_flag.set()
    auto_stop_flag.set()  # Also stop auto mode
    if session:
        asyncio.create_task(session.close())
    sys.exit(0)

async def main():
    global session
    
    log_info("=== OCTRA CLI STARTING ===")
    log_info(f"Timestamp: {datetime.now()}")
    print(f"{c['g']}Starting OCTRA CLI...{c['r']}")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"{c['c']}Loading wallet...{c['r']}")
    if not ld():
        log_error("main", Exception("Wallet loading failed"), "wallet.json error")
        print(f"{c['R']}[!] wallet.json error{c['r']}")
        sys.exit("[!] wallet.json error")
    if not addr:
        log_error("main", Exception("No address configured"), "wallet.json not configured")
        print(f"{c['R']}[!] wallet.json not configured{c['r']}")
        sys.exit("[!] wallet.json not configured")
    
    log_info(f"Wallet loaded successfully. Address: {addr}")
    log_info(f"RPC endpoint: {rpc}")
    print(f"{c['g']}✓ Wallet loaded: {addr}{c['r']}")
    print(f"{c['c']}✓ RPC: {rpc}{c['r']}")
    
    # Show user where logs are being written
    log_filename = f"octra_auto_mode_{datetime.now().strftime('%Y%m%d')}.log"
    print(f"🔧 Debug logging enabled: {log_filename}")
    print("📝 Auto mode errors and activity will be logged to this file")
    
    try:
        log_info("Fetching initial balance and history...")
        print(f"{c['c']}Fetching balance and history...{c['r']}")
        try:
            await st()
            log_info("Balance fetched successfully")
            print(f"{c['g']}✓ Balance fetched{c['r']}")
        except Exception as e:
            log_error("initialization", e, "Failed to fetch balance")
            print(f"{c['R']}Failed to fetch balance: {str(e)}{c['r']}")
            await asyncio.sleep(3)
        
        try:
            await gh()
            log_info("Transaction history fetched successfully")
            print(f"{c['g']}✓ History fetched{c['r']}")
        except Exception as e:
            log_error("initialization", e, "Failed to fetch transaction history")
            print(f"{c['R']}Failed to fetch transaction history: {str(e)}{c['r']}")
            await asyncio.sleep(3)
        
        log_info("Initialization completed, starting main loop")
        print(f"{c['g']}✓ Initialization complete{c['r']}")
        print(f"{c['B']}Starting main interface...{c['r']}")
        
        while not stop_flag.is_set():
            try:
                cmd = await scr()
                log_info(f"User input received: '{cmd}'")
            except Exception as e:
                log_error("main_loop", e, "Error in screen/input handling")
                print(f"{c['R']}Screen/input error: {str(e)}{c['r']}")
                await asyncio.sleep(2)
                continue
            
            if cmd == '1':
                await tx()
            elif cmd == '2':
                global lu, lh
                lu = lh = 0
                await st()
                await gh()
            elif cmd == '3':
                await multi()
            elif cmd == '4':
                await encrypt_balance_ui()
            elif cmd == '5':
                await decrypt_balance_ui()
            elif cmd == '6':
                await private_transfer_ui()
            elif cmd == '7':
                await claim_transfers_ui()
            elif cmd == '8':
                await exp()
            elif cmd == '9':
                h.clear()
                lh = 0
            elif cmd in ['a', 'A']:
                try:
                    log_info(f"User pressed 'a' to toggle auto mode. Current state: {auto_mode}")
                    if auto_mode:
                        await stop_auto_mode()
                    else:
                        await start_auto_mode()
                except Exception as e:
                    # Show error and continue instead of crashing
                    error_msg = f"Auto mode error: {str(e)}"
                    print(f"\033[41m\033[37m{error_msg}\033[0m")
                    log_error("main_auto_toggle", e, "User pressed 'a' to toggle auto mode")
                    print("Check octra_auto_mode_*.log for detailed error information")
                    await asyncio.sleep(5)  # Give more time to read the error
            elif cmd in ['s', 'S']:
                try:
                    log_info("User pressed 's' to view auto status")
                    await auto_status()
                except Exception as e:
                    error_msg = f"Status error: {str(e)}"
                    print(f"\033[41m\033[37m{error_msg}\033[0m")
                    log_error("main_auto_status", e, "User pressed 's' to view status")
                    await asyncio.sleep(3)
            elif cmd in ['0', 'q', '']:
                log_info("User requested exit")
                break
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        log_info("Application interrupted by user (Ctrl+C)")
    except Exception as e:
        # Show the actual error for debugging
        print(f"\033[41m\033[37mError: {str(e)}\033[0m")
        log_error("main", e, "Fatal error in main loop")
        await asyncio.sleep(3)
    finally:
        log_info("=== CLEANUP STARTING ===")
        # Stop auto mode if running
        if auto_mode:
            log_info("Stopping auto mode during cleanup")
            await stop_auto_mode()
        
        if session:
            log_info("Closing HTTP session")
            await session.close()
        
        log_info("Shutting down executor")
        executor.shutdown(wait=False)
        log_info("=== OCTRA CLI SHUTDOWN COMPLETE ===")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    finally:
        cls()
        print(f"{c['r']}")
        os._exit(0)