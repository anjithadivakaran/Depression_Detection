import torch
import torch.nn as nn
from torchvision import models
from transformers import  AutoModel
from utils import config



class DepressionClassifier(nn.Module):
    def __init__(self, text_hidden=768, img_hidden=512, fusion_hidden=256):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(config.TEXT_MODEL)

        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove final fc layer
        self.img_fc = nn.Linear(512, img_hidden)  

        self.fusion = nn.GRU(text_hidden + img_hidden, fusion_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden // 2, 1)
        )

    def forward(self, input_ids, attn_mask, images):
        B, T, _ = input_ids.size()
        input_ids = input_ids.view(B*T, -1)
        attn_mask = attn_mask.view(B*T, -1)
        images = images.view(B*T, 3, config.IMG_SIZE, config.IMG_SIZE)

        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state[:, 0]  # [B*T, 768]
        img_feat = self.image_encoder(images).squeeze(-1).squeeze(-1)  
        img_feat = self.img_fc(img_feat)  

        combined = torch.cat([text_feat, img_feat], dim=1).view(B, T, -1)
        _, h_n = self.fusion(combined)
        logits = self.classifier(h_n.squeeze(0))
        return logits.squeeze(1)