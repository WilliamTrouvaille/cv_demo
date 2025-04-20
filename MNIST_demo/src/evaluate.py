import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

from src.train import CNN
from src.utils import Loader

# 确保 docs 文件夹存在
os.makedirs(Loader.get_path("docs"), exist_ok=True)


# 加载模型
def load_model(model_path, device):
    """加载训练好的模型"""
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()  # 设置为评估模式
    return model


# 加载测试数据集
def load_test_data():
    """加载MNIST测试数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = torchvision.datasets.MNIST(
        root=Loader.get_path("data/raw"),
        train=False,
        download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1000,
        shuffle=False,
        num_workers=2
    )
    return test_loader


# 评测模型
def evaluate_model(model, test_loader, device):
    """评测模型性能"""
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_score)


# 保存分类报告
def save_classification_report(y_true, y_pred, file_path):
    """保存分类报告到文件"""
    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)])
    with open(file_path, 'w') as f:
        f.write(report)


# 保存混淆矩阵
def save_confusion_matrix(y_true, y_pred, file_path, classes):
    """保存混淆矩阵为图片"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    plt.close()


# 保存ROC曲线
def save_roc_curve(y_true, y_score, file_path, classes):
    """保存ROC曲线为图片"""
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(file_path)
    plt.close()


# 保存PR曲线
def save_pr_curve(y_true, y_score, file_path, classes):
    """保存PR曲线为图片"""
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_true == i, y_score[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(file_path)
    plt.close()


# 主函数
def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_path = Loader.get_path("models/mnist_cnn.pth")
    model = load_model(model_path, device)

    # 加载测试数据
    test_loader = load_test_data()

    # 评测模型
    y_true, y_pred, y_score = evaluate_model(model, test_loader, device)

    # 保存分类报告
    report_path = Loader.get_path("docs/classification_report.txt")
    save_classification_report(y_true, y_pred, report_path)

    # 保存混淆矩阵
    cm_path = Loader.get_path("docs/confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, cm_path, classes=[str(i) for i in range(10)])

    # 保存ROC曲线
    roc_path = Loader.get_path("docs/roc_curve.png")
    save_roc_curve(y_true, y_score, roc_path, classes=[str(i) for i in range(10)])

    # 保存PR曲线
    pr_path = Loader.get_path("docs/pr_curve.png")
    save_pr_curve(y_true, y_score, pr_path, classes=[str(i) for i in range(10)])

    print("评测结果已保存到 docs 文件夹下。")


if __name__ == "__main__":
    main()
