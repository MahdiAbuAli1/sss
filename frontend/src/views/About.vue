<template>
  <div class="about">
    <div class="max-w-4xl mx-auto">
      <h1 class="text-3xl font-bold text-gray-900 mb-8 text-center">حول المشروع</h1>
      
      <div class="bg-white rounded-lg shadow-sm p-8 mb-8">
        <h2 class="text-2xl font-semibold mb-4">الهدف من المشروع</h2>
        <p class="text-gray-600 mb-6">
          هذا المشروع يُعتبر قالباً أساسياً لمشاريع Vue.js 3 الحديثة، مُعدّ خصيصاً للمطورين العرب
          الذين يريدون البدء بمشروع جديد باستخدام أحدث التقنيات.
        </p>
        
        <h2 class="text-2xl font-semibold mb-4">التقنيات المستخدمة</h2>
        <div class="grid md:grid-cols-2 gap-6">
          <div>
            <h3 class="font-semibold text-lg mb-2">Frontend Framework</h3>
            <ul class="list-disc list-inside text-gray-600 space-y-1">
              <li>Vue.js 3 مع Composition API</li>
              <li>TypeScript للتطوير الآمن</li>
              <li>Vue Router 4 للتنقل</li>
            </ul>
          </div>
          <div>
            <h3 class="font-semibold text-lg mb-2">Build Tools & Styling</h3>
            <ul class="list-disc list-inside text-gray-600 space-y-1">
              <li>Vite للبناء السريع</li>
              <li>Tailwind CSS للتصميم</li>
              <li>PostCSS & Autoprefixer</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 class="text-xl font-semibold text-blue-900 mb-3">هيكل المشروع</h3>
        <div class="font-mono text-sm bg-white p-4 rounded border overflow-x-auto">
          <pre>src/
├── assets/
│   └── css/
│       └── main.css
├── components/
├── router/
│   └── index.ts
├── services/
├── types/
├── views/
│   ├── Home.vue
│   └── About.vue
├── App.vue
└── main.ts</pre>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// صفحة حول المشروع
</script>