import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from vbs_nn import VertexByteStream
#from model import GPTConfig 
#from model import GPT # Used nanoGPT to compare

config = GPTConfig(
    block_size = 80, 
    n_layer = 4,
    n_head = 8,
    n_embd = 256,    
    bias = False,    
    vocab_size=256
)


CURRENT_STEP = 0
GLOBAL_MODEL = None
GLOBAL_OPTIMIZER = None
GLOBAL_SEQ_LEN = 0


def decode_to_str(tensor):
    if torch.is_tensor(tensor):
        tokens = tensor.detach().cpu().flatten().tolist()
    else:
        tokens = tensor
        
    res = []
    for t in tokens:
        try:
            val = int(t)
            if 32 <= val <= 126:
                res.append(chr(val))
            else:
                res.append("░")
        except:
            res.append("?") 
            
    return "".join(res)
        
def save_checkpoint(model, optimizer, step, loss, context_len, filename="needle_checkpoint.pth"):
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'context_len': context_len
    }
    torch.save(checkpoint, filename)
    print(f"\n[v] Checkpoint saved: {filename} (Step {step}, Context {context_len})")

def load_checkpoint(model, optimizer, filename="needle_checkpoint.pth"):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("[!] Optimizer state couldn't be retrieved from the checkpoint.)")
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        context_len = checkpoint.get('context_len', 0)
        print(f"[^] Checkpoint retrieved, continue from Step {step}, Context {context_len}")
        return step, context_len
    print("Starting from scratch")
    return 0, None

def test_synthetic_niah(model, optimizer, iterations=1000, seq_len=1024, start_step=0):
    global CURRENT_STEP
    device = next(model.parameters()).device
    model.train()
    criterion = nn.CrossEntropyLoss()

    print(f"--- Starting ultimate Synthetic Needle Retrieval seq_len - ({seq_len} bytes) ---")
    successed = 0
    total_time = 0;
    start_time = time.time()
    step_start = time.time()

    for i in range(start_step, start_step + iterations):
        CURRENT_STEP = i 
       
        digits = "".join([str(random.randint(0, 9)) for _ in range(5)])
        needle_str = f" KEY:{digits} "
        needle_bytes = list(needle_str.encode('utf-8'))
        
        trigger_str = " QUESTION: What is the KEY? ANSWER:"
        trigger_bytes = list(trigger_str.encode('utf-8'))
        
        answer_len = 5

        fixed_len = len(needle_bytes) + len(trigger_bytes) + answer_len
        haystack_len = seq_len - fixed_len
        if haystack_len < 0:
            print(f"Error: seq_len {seq_len} is too small")
            break

        haystack = torch.randint(33, 126, (haystack_len,), dtype=torch.long).to(device)

        pos = random.randint(0, haystack_len)

        input_seq = torch.cat([
            haystack[:pos],
            torch.tensor(needle_bytes, dtype=torch.long).to(device),
            haystack[pos:],
            torch.tensor(trigger_bytes, dtype=torch.long).to(device),
            torch.zeros(answer_len, dtype=torch.long).to(device)
        ])
        answer_targets = torch.tensor(list(digits.encode('utf-8')), dtype=torch.long).to(device)
        needle_digits = digits
        # 2. Forward / Backward
        optimizer.zero_grad()
        #output, _ = model(input_seq.unsqueeze(0))
        output = model(input_seq.unsqueeze(0))
        logits = output[0] if isinstance(output, (tuple, list)) else output
        
        loss = criterion(logits[0, -answer_len:, :], answer_targets)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            with torch.no_grad():
                vramR = torch.cuda.memory_allocated() / 1024 ** 2
                vramA = torch.cuda.memory_reserved() / 1024 ** 2
                time_for_steps = time.time() - step_start
                step_start = time.time()
                total_time = time.time() - start_time

                trgt = decode_to_str(answer_targets)
                print(f"Answer targets {trgt}")
                pred = torch.argmax(logits[0, -5:, :], dim=-1)
                pred_str = "".join([chr(c.item()) for c in pred])
                print(f"Step {i} | Loss: {loss.item():.4f} | Target: {needle_digits} | Pred: {pred_str} | VRAM(model): {vramR:.0f}MB | VRAM(total): {vramA:.0f}MB")
                print(f"Runtimes  100 steps : {time_for_steps:.3f}s Total {total_time:.3f}s")
                full_text = decode_to_str(input_seq)
                full_text_len = len(full_text) 
                if full_text_len > 200:
                    print(f"\n\n[DEBUG STEP {i}]")
                    print(f"Start: {full_text[:50]}...")
                    if 'pos' in locals():
                        print(f"Needle area ({pos-20}): ...{full_text[max(0, pos-20):pos+40]}...")
                    print(f"End (Target area) ({full_text_len-50}): ...{full_text[-50:]}")
                else:
                    print(f"\n[DEBUG STEP {i}] Full Input: {full_text}")
        if loss.item() < 0.05:
            successed+=1
            if (successed > 100):
                save_checkpoint(model, optimizer, i, loss.item(), seq_len, f"needle_success_{seq_len}.pth")
                seq_len = seq_len << 1
                print("=" * 80)
                print(f"Increasing context length, new context {seq_len}")
                print("=" * 80)
                successed = 0
        else:
            successed-=1
            successed = 0 if successed < 0 else successed
#            sys.exit(0)

        
    model.eval()
    with torch.no_grad():
        test_needle = "98765"
        t_prefix = torch.tensor(list(f"KEY:{test_needle} ".encode('utf-8')), dtype=torch.long).to(device)
        t_suffix = torch.tensor(list(" ANSWER:".encode('utf-8')), dtype=torch.long).to(device)
        t_haystack = torch.randint(33, 126, (seq_len - len(t_prefix) - len(t_suffix) - 5,), dtype=torch.long).to(device)
        t_input = torch.cat([t_prefix, t_haystack, t_suffix, torch.zeros(5, dtype=torch.long).to(device)]).unsqueeze(0)
        
        out = model(t_input)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        res_tokens = torch.argmax(logits[0, -5:, :], dim=-1)
        res_str = "".join([chr(c.item()) for c in res_tokens])
        
        print(f"\n--- FINAL TEST ---")
        print(f"Needle: {test_needle}")
        print(f"Model retrieve: {res_str}")

vocab_size = 256
D_MODEL = 128
LEVELS = 15

GLOBAL_MODEL = VertexByteStream(vocab_size, D_MODEL, LEVELS).cuda()
#GLOBAL_MODEL = GPT(config).cuda()
GLOBAL_OPTIMIZER = optim.Adam(GLOBAL_MODEL.parameters(), lr=1e-4)


start_from, last_context = load_checkpoint(GLOBAL_MODEL, GLOBAL_OPTIMIZER)
step=0
start_step=0
seq_len = 64
GLOBAL_SEQ_LEN = seq_len

#new_lr = 1e-4
#for param_group in GLOBAL_OPTIMIZER.param_groups:
#    param_group['lr'] = new_lr

try:
    test_synthetic_niah(GLOBAL_MODEL, GLOBAL_OPTIMIZER, 100000, GLOBAL_SEQ_LEN, start_step=start_from)
except KeyboardInterrupt:
    print("\n[!] Training stopped by Contorl-C.")
finally:
    save_checkpoint(GLOBAL_MODEL, GLOBAL_OPTIMIZER, CURRENT_STEP, 0.0, GLOBAL_SEQ_LEN)
 

