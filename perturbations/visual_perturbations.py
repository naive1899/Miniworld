import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class PerturbationManager:
    """Менеджер визуальных помех."""
    
    def __init__(self):
        self.current = 'none'
        self.severity = 0.5
        
        self._methods = {
            'none': lambda img, sev: img,
            'gaussian': self._gaussian_noise,
            'impulse': self._impulse_noise,
            'blur': self._gaussian_blur,
            'color': self._color_jitter,
            'occlusion': self._occlusion,
            'fog': self._fog,
        }
        self._randomizable = list(self._methods.keys())[1:]
    
    def _gaussian_noise(self, img, sev):
        noise = np.random.normal(0, sev * 50, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    def _impulse_noise(self, img, sev):
        noisy = img.copy()
        prob = sev * 0.3
        salt = np.random.random(img.shape[:2]) < prob / 2
        pepper = np.random.random(img.shape[:2]) < prob / 2
        noisy[salt] = 255
        noisy[pepper] = 0
        return noisy
    
    def _gaussian_blur(self, img, sev):
        pil_img = Image.fromarray(img)
        radius = sev * 5
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.array(blurred)
    
    def _color_jitter(self, img, sev):
        pil_img = Image.fromarray(img)
        
        # Яркость
        enhancer = ImageEnhance.Brightness(pil_img)
        brightness = np.random.uniform(1 - sev, 1 + sev)
        pil_img = enhancer.enhance(brightness)
        
        # Контраст
        enhancer = ImageEnhance.Contrast(pil_img)
        contrast = np.random.uniform(1 - sev, 1 + sev)
        pil_img = enhancer.enhance(contrast)
        
        # Цвет
        enhancer = ImageEnhance.Color(pil_img)
        color = np.random.uniform(1 - sev, 1 + sev)
        pil_img = enhancer.enhance(color)
        
        return np.array(pil_img)
    
    def _occlusion(self, img, sev):
        h, w = img.shape[:2]
        size = int(min(h, w) * sev * 0.5)
        if size < 1:
            return img
        
        x = np.random.randint(0, w - size + 1)
        y = np.random.randint(0, h - size + 1)
        
        img_copy = img.copy()
        img_copy[y:y+size, x:x+size] = 0
        return img_copy
    
    def _fog(self, img, sev):
        h, w = img.shape[:2]
        y_coords, x_coords = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        fog_intensity = (dist / max_dist * sev)[:, :, np.newaxis]
        fog_color = np.ones_like(img) * 255
        
        return (img * (1 - fog_intensity) + fog_color * fog_intensity).astype(np.uint8)
    
    def apply(self, img):
        return self._methods[self.current](img, self.severity)
    
    def set(self, name, severity=None):
        if name not in self._methods:
            raise ValueError(f"Unknown: {name}")
        self.current = name
        if severity is not None:
            self.severity = np.clip(severity, 0.0, 1.0)
    
    def randomize(self):
        import random
        name = random.choice(self._randomizable)
        sev = random.uniform(0.1, 0.7)
        self.set(name, sev)
        return name, sev
    
    def get_available(self):
        return list(self._methods.keys())