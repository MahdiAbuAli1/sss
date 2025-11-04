// تعريفات أنواع أساسية للمشروع

export interface User {
  id: number
  name: string
  email: string
  avatar?: string
}

export interface ApiResponse<T> {
  data: T
  message: string
  status: number
}

export interface PaginatedResponse<T> {
  data: T[]
  total: number
  page: number
  limit: number
}

export interface NavItem {
  name: string
  path: string
  icon?: string
  children?: NavItem[]
}