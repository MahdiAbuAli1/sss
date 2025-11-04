<template>
  <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow duration-200">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div class="flex items-center">
        <div 
          class="flex-shrink-0 p-3 rounded-lg"
          :class="iconBackgroundClass"
        >
          <component 
            :is="iconComponent" 
            class="w-6 h-6"
            :class="iconColorClass"
          />
        </div>
      </div>
      
      <!-- Period Selector (Optional) -->
      <div v-if="showPeriodSelector" class="flex items-center space-x-2 space-x-reverse">
        <select 
          v-model="selectedPeriod"
          @change="$emit('period-change', selectedPeriod)"
          class="text-xs border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="today">اليوم</option>
          <option value="week">هذا الأسبوع</option>
          <option value="month">هذا الشهر</option>
          <option value="year">هذا العام</option>
        </select>
      </div>
    </div>

    <!-- Content -->
    <div class="mt-4">
      <div class="flex items-baseline">
        <h3 class="text-2xl font-bold text-gray-900">
          {{ formattedValue }}
        </h3>
        <span v-if="unit" class="mr-2 text-sm text-gray-500">
          {{ unit }}
        </span>
      </div>

      <!-- Title -->
      <p class="mt-1 text-sm text-gray-600">
        {{ title }}
      </p>

      <!-- Description -->
      <p v-if="description" class="mt-1 text-xs text-gray-500">
        {{ description }}
      </p>

      <!-- Trend -->
      <div v-if="showTrend" class="mt-3 flex items-center">
        <div 
          class="flex items-center text-sm font-medium"
          :class="trendColorClass"
        >
          <component 
            :is="trendIcon"
            class="w-4 h-4 ml-1"
          />
          {{ trendValue }}
        </div>
        <span class="mr-2 text-sm text-gray-500">
          مقارنة بالفترة السابقة
        </span>
      </div>

      <!-- Progress Bar (Optional) -->
      <div v-if="showProgress" class="mt-4">
        <div class="flex items-center justify-between text-xs text-gray-500 mb-1">
          <span>التقدم</span>
          <span>{{ progress }}%</span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-2">
          <div 
            class="h-2 rounded-full transition-all duration-500"
            :class="progressColorClass"
            :style="{ width: `${Math.min(progress, 100)}%` }"
          ></div>
        </div>
      </div>

      <!-- Action Button (Optional) -->
      <div v-if="actionButton" class="mt-4">
        <button 
          @click="$emit('action-click')"
          class="text-sm font-medium text-blue-600 hover:text-blue-800 transition-colors duration-200"
        >
          {{ actionButton }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

// Props
interface Props {
  title: string
  value: number | string
  unit?: string
  description?: string
  icon?: string
  iconColor?: 'blue' | 'green' | 'yellow' | 'red' | 'purple' | 'gray'
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
  showTrend?: boolean
  showProgress?: boolean
  progress?: number
  showPeriodSelector?: boolean
  actionButton?: string
  prefix?: string
  suffix?: string
  decimals?: number
}

const props = withDefaults(defineProps<Props>(), {
  iconColor: 'blue',
  showTrend: true,
  showProgress: false,
  showPeriodSelector: false,
  decimals: 0,
  prefix: '',
  suffix: ''
})

// Emits
const emit = defineEmits<{
  'period-change': [period: string]
  'action-click': []
}>()

// State
const selectedPeriod = ref('month')

// Computed
const formattedValue = computed(() => {
  if (typeof props.value === 'number') {
    return `${props.prefix}${props.value.toLocaleString('ar-SA', { 
      minimumFractionDigits: props.decimals,
      maximumFractionDigits: props.decimals 
    })}${props.suffix}`
  }
  return props.value
})

const iconComponent = computed(() => {
  const icons: Record<string, any> = {
    users: {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z"/></svg>`
    },
    revenue: {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`
    },
    orders: {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"/></svg>`
    },
    products: {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/></svg>`
    },
    chart: {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>`
    },
    activity: {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>`
    }
  }
  return icons[props.icon || 'chart']
})

const iconColorClass = computed(() => {
  const colors = {
    blue: 'text-blue-600',
    green: 'text-green-600',
    yellow: 'text-yellow-600',
    red: 'text-red-600',
    purple: 'text-purple-600',
    gray: 'text-gray-600'
  }
  return colors[props.iconColor]
})

const iconBackgroundClass = computed(() => {
  const colors = {
    blue: 'bg-blue-100',
    green: 'bg-green-100',
    yellow: 'bg-yellow-100',
    red: 'bg-red-100',
    purple: 'bg-purple-100',
    gray: 'bg-gray-100'
  }
  return colors[props.iconColor]
})

const trendIcon = computed(() => {
  if (!props.showTrend) return null
  
  if (props.trend === 'up') {
    return {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 17l9.2-9.2M17 17V7H7"/></svg>`
    }
  } else if (props.trend === 'down') {
    return {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 7L7.8 16.2M7 7v10h10"/></svg>`
    }
  } else {
    return {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/></svg>`
    }
  }
})

const trendColorClass = computed(() => {
  if (!props.showTrend) return ''
  
  const colors = {
    up: 'text-green-600',
    down: 'text-red-600',
    neutral: 'text-gray-600'
  }
  return colors[props.trend || 'neutral']
})

const progressColorClass = computed(() => {
  const colors = {
    blue: 'bg-blue-600',
    green: 'bg-green-600',
    yellow: 'bg-yellow-600',
    red: 'bg-red-600',
    purple: 'bg-purple-600',
    gray: 'bg-gray-600'
  }
  return colors[props.iconColor]
})
</script>

<style scoped>
/* تحسينات إضافية للتأثيرات */
.transition-all {
  transition-property: all;
  transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}
</style>