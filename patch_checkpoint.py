import torch, sys
sys.path.insert(0, 'src')

ckpt = torch.load('checkpoints/ids_v14_model.pth', map_location='cpu', weights_only=False)

# enc[4] = Linear(mid, mid//2), shape = (mid//2, mid)
# So enc[4].weight.shape[1] = mid = ae_hidden
correct_ae_hidden = ckpt['model_state_dict']['ae.enc.4.weight'].shape[1]
correct_hidden    = ckpt['model_state_dict']['backbone.input_proj.0.weight'].shape[0]

print(f"ae_hidden cu (sai): {ckpt.get('ae_hidden')}")
print(f"ae_hidden moi (dung): {correct_ae_hidden}")
print(f"hidden: {correct_hidden}")

ckpt['ae_hidden'] = correct_ae_hidden
ckpt['hidden']    = correct_hidden

torch.save(ckpt, 'checkpoints/ids_v14_model.pth')
print("Da patch xong! Checkpoint da duoc cap nhat.")
