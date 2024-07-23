import type { Metadata } from 'next'
import { Inter, Roboto_Mono } from 'next/font/google'
import './globals.css'
import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/sonner"

const inter = Inter({ subsets: ['latin'] })
const rm = Roboto_Mono({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Security Camera AI',
  description: '24x7 surveillance',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system">
          {children}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}
