// المكونات المشتركة المصدرة
export { default as Sidebar } from './Sidebar.vue'
export { default as StatCard } from './StatCard.vue'
export { default as DataTable } from './DataTable.vue'
export { default as Modal } from './Modal.vue'

// أنواع البيانات (Types)
export interface StatCardProps {
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

export interface DataTableColumn {
  key: string
  label: string
  type?: 'text' | 'html' | 'badge' | 'actions'
  formatter?: (value: any, row: any) => string
  sortable?: boolean
  actions?: Array<{
    label: string
    handler: (row: any, index: number) => void
  }>
}

export interface DataTableFilter {
  key: string
  label: string
  type: 'text' | 'select' | 'date'
  options?: Array<{ value: string | number; label: string }>
  placeholder?: string
}

export interface DataTableAction {
  key: string
  label: string
  class?: string
  icon?: any
  handler: (row: any, index: number) => void
}

export interface ModalProps {
  show: boolean
  title?: string
  subtitle?: string
  content?: string
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'full'
  closable?: boolean
  closeOnOverlay?: boolean
  showHeader?: boolean
  showFooter?: boolean
  showConfirmButton?: boolean
  showCancelButton?: boolean
  confirmText?: string
  cancelText?: string
  loadingText?: string
  confirmButtonType?: 'primary' | 'success' | 'warning' | 'danger' | 'info'
  icon?: string
  iconType?: 'info' | 'success' | 'warning' | 'error'
  persistent?: boolean
  loading?: boolean
  maxHeight?: string
  width?: string
}