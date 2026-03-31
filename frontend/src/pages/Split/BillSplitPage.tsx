import { useState, useRef, useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { useContactsStore } from '../../store/contacts.store'
import { uploadReceipt, scanReceipt } from '../../api/receipts'
import { createBillSplit, updateBillSplit, listBillSplits, deleteBillSplit } from '../../api/billSplits'
import type { BillSplitResponse } from '../../api/billSplits'
import { getMe } from '../../api/auth'
import { useAuthStore } from '../../store/auth.store'
import { avatarColor } from '../../utils/avatar'
import { IconPencil, IconTrash, iconBtnEdit, iconBtnDelete } from '../../utils/icons'

// ─── Types ────────────────────────────────────────────────────────────────────

type Person = { id: string; name: string; friendId?: string }

type ItemState = {
  id: string
  name: string
  quantity: number
  unitPrice: number
  discountAmount: number
  isTaxable: boolean
  taxRate: number
  assignedTo: string[]
  isTaxLine: boolean
  isTipLine: boolean
}

type ReceiptState = {
  id: string
  backendReceiptId?: string   // the real DB receipt UUID after scanning
  merchantName: string
  date: string                // YYYY-MM-DD
  paidById: string
  items: ItemState[]
  scanning: boolean
  error: string
  ocr_total?: number
  isManual?: boolean
}

type Step = 'people' | 'receipts' | 'review' | 'summary'

const DEFAULT_TAX = 8.25

// ─── Helpers ──────────────────────────────────────────────────────────────────

function genId() { return Math.random().toString(36).slice(2, 10) }

function itemTotal(item: ItemState): number {
  const base = item.unitPrice * item.quantity
  const afterDiscount = Math.max(0, base - (item.discountAmount || 0))
  const tax = item.isTaxable ? afterDiscount * (item.taxRate / 100) : 0
  return afterDiscount + tax
}


// Tax portion covered by per-item checkboxes
function perItemTaxCovered(r: ReceiptState): number {
  return r.items
    .filter(i => !i.isTaxLine && i.isTaxable)
    .reduce((s, i) => {
      const base = i.unitPrice * i.quantity
      const afterDiscount = Math.max(0, base - (i.discountAmount || 0))
      return s + afterDiscount * (i.taxRate / 100)
    }, 0)
}

// Effective TAX line amount = original tax - already covered by checkboxes
function effectiveTaxLineAmount(r: ReceiptState, item: ItemState): number {
  if (!item.isTaxLine) return itemTotal(item)
  const original = item.unitPrice * item.quantity
  const covered = perItemTaxCovered(r)
  return Math.max(0, original - covered)
}

// Items subtotal (non-tax-line items, including any per-item tax added via checkbox)
function itemsSubtotal(r: ReceiptState): number {
  return r.items.filter(i => !i.isTaxLine && !i.isTipLine).reduce((s, i) => s + itemTotal(i), 0)
}

// Effective tax total (tax lines after subtracting per-item coverage)
function effectiveTaxTotal(r: ReceiptState): number {
  return r.items.filter(i => i.isTaxLine).reduce((s, i) => s + effectiveTaxLineAmount(r, i), 0)
}

// Tip total
function tipTotal(r: ReceiptState): number {
  return r.items.filter(i => i.isTipLine).reduce((s, i) => s + itemTotal(i), 0)
}

// Total tax embedded in per-item taxable amounts (for display only)
function embeddedTaxTotal(r: ReceiptState): number {
  return r.items
    .filter(i => !i.isTaxLine && !i.isTipLine && i.isTaxable)
    .reduce((s, i) => {
      const base = Math.max(0, i.unitPrice * i.quantity - (i.discountAmount || 0))
      return s + base * (i.taxRate / 100)
    }, 0)
}

// Grand effective total = items + effective tax + tips
function receiptEffectiveTotal(r: ReceiptState): number {
  return itemsSubtotal(r) + effectiveTaxTotal(r) + tipTotal(r)
}

function receiptItemsTotal(r: ReceiptState) {
  return receiptEffectiveTotal(r)
}

function isTallied(r: ReceiptState) {
  if (!r.ocr_total) return true
  return Math.abs(receiptEffectiveTotal(r) - r.ocr_total) < 0.02
}

function hasDoubleTax(_r: ReceiptState) { return false }

// ─── Main Component ───────────────────────────────────────────────────────────

export default function BillSplitPage() {
  const [step, setStep] = useState<Step>('people')
  const [people, setPeople] = useState<Person[]>([])
  const [nameInput, setNameInput] = useState('')
  const [receipts, setReceipts] = useState<ReceiptState[]>([])
  const [peopleOpen, setPeopleOpen] = useState(false)
  const [quickName, setQuickName] = useState('')
  const [savedSplitId, setSavedSplitId] = useState<string | null>(null)
  const [showSavedSplits, setShowSavedSplits] = useState(false)
  const [manualInputs, setManualInputs] = useState<Record<string, { name: string; price: string }>>({})
  const [editingItemId, setEditingItemId] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()

  const { contacts, groups: contactGroups } = useContactsStore()
  const [showDropdown, setShowDropdown] = useState(false)
  useAuthStore(s => s.token) // keep store subscribed for auth
  const { data: me } = useQuery({ queryKey: ['me'], queryFn: getMe })

  // ── Saved splits ────────────────────────────────────────────────────────────
  const savedSplitsQuery = useQuery({
    queryKey: ['bill-splits'],
    queryFn: listBillSplits,
  })

  const buildSplitState = () => ({
    people,
    receipts: receipts.map(r => ({
      id: r.id,
      backendReceiptId: r.backendReceiptId,
      merchantName: r.merchantName,
      paidById: r.paidById,
      ocr_total: r.ocr_total,
      items: r.items,
    })),
  })

  const loadSplit = (saved: BillSplitResponse) => {
    const s = saved.state as any
    setPeople(s.people || [])
    setReceipts((s.receipts || []).map((r: any) => ({
      ...r,
      date: r.date || new Date().toISOString().split('T')[0],
      scanning: false,
      error: '',
    })))
    setSavedSplitId(saved.id)
    setStep('summary')
    setShowSavedSplits(false)
  }

  const saveSplit = useMutation({
    mutationFn: async () => {
      const merchantNames = receipts.map(r => r.merchantName || 'Receipt').filter(Boolean)
      const title = merchantNames.length ? merchantNames.join(', ') : 'Bill Split'
      const state = buildSplitState()
      if (savedSplitId) {
        return updateBillSplit(savedSplitId, title, state)
      } else {
        return createBillSplit(title, state)
      }
    },
    onSuccess: (data) => {
      setSavedSplitId(data.id)
      queryClient.invalidateQueries({ queryKey: ['bill-splits'] })
    },
  })

  const deleteSplit = useMutation({
    mutationFn: (id: string) => deleteBillSplit(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bill-splits'] }),
  })

  // Auto-load from ?load=id URL param (coming from Activity page)
  useEffect(() => {
    const loadId = searchParams.get('load')
    if (!loadId || !savedSplitsQuery.data) return
    const found = savedSplitsQuery.data.find(s => s.id === loadId)
    if (found) {
      loadSplit(found)
      setSearchParams({}, { replace: true })
    }
  }, [searchParams, savedSplitsQuery.data])

  // Listen for "New split" event dispatched from sidebar
  useEffect(() => {
    const handler = () => {
      setPeople([]); setReceipts([])
      setSavedSplitId(null); setStep('people'); saveSplit.reset()
    }
    window.addEventListener('splitit:new-split', handler)
    return () => window.removeEventListener('splitit:new-split', handler)
  }, [])

  // ── People helpers ──────────────────────────────────────────────────────────
  const addPerson = (name: string, contactId?: string) => {
    const trimmed = name.trim()
    if (!trimmed) return
    // Dedupe by contactId if provided, otherwise by name
    if (contactId && people.find(p => p.id === contactId)) return
    if (!contactId && people.find(p => p.name.toLowerCase() === trimmed.toLowerCase())) return
    setPeople(prev => [...prev, { id: contactId || genId(), name: trimmed, friendId: contactId }])
    setNameInput('')
  }

  // Contacts not yet added as people
  // "You" suggestion — the logged-in user shown at top of dropdown
  const meSuggestion = me && !people.find(p => p.id === me.id) &&
    (nameInput.trim() === '' || me.display_name.toLowerCase().includes(nameInput.toLowerCase()))
    ? [{ id: me.id, name: me.display_name, isMe: true }]
    : []

  const contactSuggestions = [
    ...meSuggestion,
    ...contacts.filter(c =>
      !people.find(p => p.id === c.id) &&
      (nameInput.trim() === '' || c.name.toLowerCase().includes(nameInput.toLowerCase()))
    ).map(c => ({ ...c, isMe: false })),
  ]

  // ── Receipt helpers ─────────────────────────────────────────────────────────
  const updateItem = (rId: string, itemId: string, changes: Partial<ItemState>) =>
    setReceipts(prev => prev.map(r =>
      r.id !== rId ? r : { ...r, items: r.items.map(i => i.id === itemId ? { ...i, ...changes } : i) }
    ))

  const togglePerson = (rId: string, itemId: string, personId: string) =>
    setReceipts(prev => prev.map(r => {
      if (r.id !== rId) return r
      return {
        ...r,
        items: r.items.map(i => {
          if (i.id !== itemId) return i
          const has = i.assignedTo.includes(personId)
          return { ...i, assignedTo: has ? i.assignedTo.filter(x => x !== personId) : [...i.assignedTo, personId] }
        })
      }
    }))

  const handleFile = async (file: File, rId: string) => {
    setReceipts(prev => prev.map(r => r.id === rId ? { ...r, scanning: true, error: '' } : r))
    try {
      const uploaded = await uploadReceipt(file)
      const scanned = await scanReceipt(uploaded.id)
      setReceipts(prev => prev.map(r => {
        if (r.id !== rId) return r
        return {
          ...r,
          scanning: false,
          backendReceiptId: scanned.id,
          merchantName: scanned.merchant_name || 'Receipt',
          date: r.date || today(),
          ocr_total: scanned.ocr_total ?? undefined,
          items: (scanned.items as any[]).map(item => ({
            id: item.id,
            name: item.name,
            quantity: parseFloat(item.quantity) || 1,
            unitPrice: parseFloat(item.unit_price) || 0,
            discountAmount: parseFloat(item.discount_amount) || 0,
            isTaxable: item.is_taxable || false,
            taxRate: parseFloat(item.tax_rate) || 0,
            assignedTo: item.is_tip_line ? people.map(p => p.id) : [],
            isTaxLine: false,  // tax is now distributed per-item, no separate line
            isTipLine: item.is_tip_line || false,
          }))
        }
      }))
    } catch (e: any) {
      const msg = e?.response?.data?.detail || 'Scan failed — try again'
      setReceipts(prev => prev.map(r => r.id === rId ? { ...r, scanning: false, error: msg } : r))
    }
  }

  const today = () => new Date().toISOString().split('T')[0]

  // Start scanning multiple files in parallel — creates one receipt slot per file
  const handleFiles = (files: File[]) => {
    if (!files.length) return
    const slots = files.map(() => ({ id: genId() }))
    setReceipts(prev => [
      ...prev,
      ...slots.map(({ id }) => ({
        id,
        merchantName: '',
        date: today(),
        paidById: people[0]?.id || '',
        items: [],
        scanning: true,
        error: '',
      }))
    ])
    slots.forEach(({ id }, i) => handleFile(files[i], id))
  }

  const addManualReceipt = () => {
    const id = genId()
    setReceipts(prev => [...prev, {
      id, merchantName: '', date: today(), paidById: people[0]?.id || '',
      items: [], scanning: false, error: '', isManual: true,
    }])
    setManualInputs(prev => ({ ...prev, [id]: { name: '', price: '' } }))
  }

  const addManualItem = (rId: string) => {
    const input = manualInputs[rId]
    if (!input?.name.trim() || !input?.price) return
    const price = parseFloat(input.price)
    if (isNaN(price) || price <= 0) return
    const itemId = genId()
    setReceipts(prev => prev.map(r => r.id !== rId ? r : {
      ...r,
      items: [...r.items, {
        id: itemId,
        name: input.name.trim(),
        quantity: 1,
        unitPrice: price,
        discountAmount: 0,
        isTaxable: false,
        taxRate: DEFAULT_TAX,
        assignedTo: [],
        isTaxLine: false,
        isTipLine: false,
      }]
    }))
    setManualInputs(prev => ({ ...prev, [rId]: { name: '', price: '' } }))
  }

  // ── Summary ─────────────────────────────────────────────────────────────────
  const summary = people.map(person => {
    let share = 0
    let paid = 0
    for (const r of receipts) {
      const rTotal = receiptEffectiveTotal(r)
      if (r.paidById === person.id) paid += rTotal
      for (const item of r.items) {
        if (!item.assignedTo.includes(person.id) || item.assignedTo.length === 0) continue
        // Use effective amount for tax lines (remaining after per-item coverage)
        const amt = item.isTaxLine ? effectiveTaxLineAmount(r, item) : itemTotal(item)
        share += amt / item.assignedTo.length
      }
    }
    return { person, share, paid, net: share - paid }
  })

  const settlements = (() => {
    const debtors = summary.filter(s => s.net > 0.01).map(s => ({ ...s })).sort((a, b) => b.net - a.net)
    const creditors = summary.filter(s => s.net < -0.01).map(s => ({ ...s })).sort((a, b) => a.net - b.net)
    const result: { from: string; to: string; amount: number }[] = []
    let i = 0, j = 0
    while (i < debtors.length && j < creditors.length) {
      const amount = Math.min(debtors[i].net, -creditors[j].net)
      if (amount > 0.01) result.push({ from: debtors[i].person.name, to: creditors[j].person.name, amount })
      debtors[i].net -= amount
      creditors[j].net += amount
      if (debtors[i].net < 0.01) i++
      if (creditors[j].net > -0.01) j++
    }
    return result
  })()

  const grandTotal = receipts.reduce((s, r) => s + receiptEffectiveTotal(r), 0)
  const STEPS: Step[] = ['people', 'receipts', 'review', 'summary']
  const stepIdx = STEPS.indexOf(step)

  return (
    <div>
      {/* Header */}
      <div className="bg-white border-b border-[#e8e8ed] px-8 py-5 sticky top-0 z-10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold text-[#1d1d1f] tracking-tight">Bill Calculator</h1>
            {savedSplitId && (
              <span className="text-xs bg-[#edfaf1] border border-[#34c759]/30 text-[#1a7a3a] px-2.5 py-1 rounded-full font-medium">
                ✓ Saved
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5">
            {STEPS.map((s, i) => (
              <div key={s} className="flex items-center gap-1.5">
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium transition-all ${
                  i < stepIdx ? 'bg-[#34c759] text-white' :
                  i === stepIdx ? 'bg-[#0071e3] text-white' :
                  'bg-[#f2f2f7] text-[#86868b]'
                }`}>{i < stepIdx ? '✓' : i + 1}</div>
                {i < 3 && <div className="w-5 h-px bg-[#d2d2d7]" />}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── PEOPLE QUICK-EDIT BAR (shown on steps after 'people') ─────────── */}
      {step !== 'people' && (
        <div className="border-b border-[#e8e8ed] bg-[#f9f9f9]">
          <div className="max-w-3xl mx-auto px-8">
            <button
              onClick={() => setPeopleOpen(o => !o)}
              className="flex items-center gap-2 py-3 text-sm text-[#3a3a3c] hover:text-[#1d1d1f] w-full text-left"
            >
              <div className="flex -space-x-1.5">
                {people.slice(0, 5).map(p => (
                  <div key={p.id} className="w-5 h-5 rounded-full text-white flex items-center justify-center text-[9px] font-bold border border-white" style={{ background: avatarColor(p.name).bg, color: avatarColor(p.name).text }}>
                    {p.name[0].toUpperCase()}
                  </div>
                ))}
              </div>
              <span className="font-medium">{people.length} {people.length === 1 ? 'person' : 'people'}</span>
              <span className="text-[#86868b]">· tap to edit</span>
              <span className="ml-auto text-[#86868b]">{peopleOpen ? '▲' : '▼'}</span>
            </button>

            {peopleOpen && (
              <div className="pb-4 space-y-3">
                <div className="flex flex-wrap gap-2">
                  {people.map(p => (
                    <div key={p.id} className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium text-white" style={{ background: avatarColor(p.name).bg, color: avatarColor(p.name).text }}>
                      {p.name}
                      <button
                        onClick={() => setPeople(prev => prev.filter(x => x.id !== p.id))}
                        className="opacity-70 hover:opacity-100 ml-0.5 leading-none"
                      >✕</button>
                    </div>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input
                    value={quickName}
                    onChange={e => setQuickName(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') { addPerson(quickName); setQuickName('') } }}
                    placeholder="Add a person..."
                    className="flex-1 border border-[#d2d2d7] rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#0071e3] bg-white"
                  />
                  <button
                    onClick={() => { addPerson(quickName); setQuickName('') }}
                    disabled={!quickName.trim()}
                    className="bg-[#0071e3] text-white px-4 py-2 rounded-xl text-sm font-medium disabled:opacity-40 hover:bg-[#0077ed] transition-colors"
                  >Add</button>
                </div>
                {(() => {
                  const quickSuggestions = [
                    ...(me && !people.find(p => p.id === me.id) ? [{ id: me.id, name: me.display_name, isMe: true }] : []),
                    ...contacts.filter(c => !people.find(p => p.id === c.id)).map(c => ({ ...c, isMe: false })),
                  ]
                  return quickSuggestions.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {quickSuggestions.map(c => (
                        <button
                          key={c.id}
                          onClick={() => addPerson(c.name, c.id)}
                          className="text-xs px-3 py-1.5 rounded-full border border-[#d2d2d7] text-[#3a3a3c] hover:border-[#0071e3] hover:text-[#0071e3] transition-colors"
                        >+ {c.name}{c.isMe ? ' (You)' : ''}</button>
                      ))}
                    </div>
                  ) : null
                })()}
              </div>
            )}
          </div>
        </div>
      )}

      <div className="px-8 py-8 max-w-3xl mx-auto">

        {/* ── STEP 1: PEOPLE ───────────────────────────────────────────────── */}
        {step === 'people' && (
          <div className="space-y-4">
            <div className="mb-6">
              <h2 className="text-2xl font-semibold text-[#1d1d1f] tracking-tight mb-1">Who's splitting?</h2>
              <p className="text-sm text-[#6e6e73]">Add everyone involved in this bill</p>
            </div>

            <div className="bg-white rounded-2xl border border-[#e8e8ed] p-5">
              {/* Name input with dropdown */}
              <div className="relative">
                <div className="flex gap-2">
                  <input
                    value={nameInput}
                    onChange={e => { setNameInput(e.target.value); setShowDropdown(true) }}
                    onFocus={() => setShowDropdown(true)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') { addPerson(nameInput); setShowDropdown(false) }
                      if (e.key === 'Escape') setShowDropdown(false)
                    }}
                    placeholder="Type a name..."
                    autoFocus
                    className="flex-1 border border-[#d2d2d7] rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#0071e3] bg-[#f9f9f9]"
                  />
                  <button
                    onClick={() => { addPerson(nameInput); setShowDropdown(false) }}
                    disabled={!nameInput.trim()}
                    className="bg-[#0071e3] text-white px-4 py-2 rounded-xl text-sm font-medium disabled:opacity-40 hover:bg-[#0077ed] transition-colors"
                  >Add</button>
                </div>

                {/* Contact dropdown — only when typing */}
                {showDropdown && nameInput.trim().length > 0 && contactSuggestions.length > 0 && (
                  <div className="absolute top-full left-0 right-10 mt-1 bg-white border border-[#e8e8ed] rounded-xl shadow-lg z-20 overflow-hidden">
                    {contactSuggestions.slice(0, 6).map(c => (
                      <button
                        key={c.id}
                        onMouseDown={e => { e.preventDefault(); addPerson(c.name, c.id); setShowDropdown(false) }}
                        className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-[#f5f5f7] transition-colors text-left"
                      >
                        <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0" style={{ background: avatarColor(c.name).bg, color: avatarColor(c.name).text }}>
                          {c.name[0].toUpperCase()}
                        </div>
                        <div className="flex items-center gap-2">
                          <p className="text-sm font-medium text-[#1d1d1f]">{c.name}</p>
                          {(c as any).isMe && <span className="text-xs bg-[#e8f1fb] text-[#0071e3] px-2 py-0.5 rounded-full font-medium">You</span>}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Group chips below input */}
              {contactGroups.length > 0 && (
                <div className="mt-3 pt-3 border-t border-[#f2f2f7]">
                  <p className="text-xs text-[#aeaeb2] mb-2">Groups</p>
                  <div className="flex flex-wrap gap-2">
                    {contactGroups.map(g => {
                      // Resolve members: check contacts first, then fall back to "me"
                      const members = g.memberIds.map(id => {
                        if (me && id === me.id) return { id: me.id, name: me.display_name }
                        return contacts.find(c => c.id === id) || null
                      }).filter(Boolean) as { id: string; name: string }[]
                      const allAdded = members.length > 0 && members.every(m => people.find(p => p.id === m.id))
                      return (
                        <button
                          key={g.id}
                          onClick={() => {
                            if (allAdded) {
                              // Deselect — only remove members NOT in any other selected group
                              const otherGroups = contactGroups.filter(og => og.id !== g.id)
                              const otherGroupMemberIds = new Set(
                                otherGroups.flatMap(og =>
                                  og.memberIds.filter(_id => {
                                    const ogMembers = og.memberIds.map(mid => {
                                      if (me && mid === me.id) return { id: me.id, name: me.display_name }
                                      return contacts.find(c => c.id === mid) || null
                                    }).filter(Boolean) as any[]
                                    return ogMembers.length > 0 && ogMembers.every(m => people.find(p => p.id === m.id))
                                  })
                                )
                              )
                              setPeople(prev => prev.filter(p =>
                                !members.find(m => m.id === p.id) || otherGroupMemberIds.has(p.id)
                              ))
                            } else {
                              // Select — add all members
                              members.forEach(m => addPerson(m.name, m.id))
                            }
                          }}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors"
                          style={{
                            borderColor: allAdded ? '#1d1d1f' : '#0071e3',
                            color: allAdded ? '#fff' : '#0071e3',
                            background: allAdded ? '#1d1d1f' : '#f0f7ff'
                          }}
                        >
                          {allAdded
                            ? <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>
                            : <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                          }
                          {g.name}
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}

              {people.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[#f2f2f7]">
                  {people.map(p => (
                    <div key={p.id} className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium text-white" style={{ background: avatarColor(p.name).bg, color: avatarColor(p.name).text }}>
                      {p.name}
                      <button onClick={() => setPeople(prev => prev.filter(x => x.id !== p.id))} className="opacity-70 hover:opacity-100 ml-0.5">✕</button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {contacts.length === 0 && (
              <p className="text-xs text-[#aeaeb2] text-center">
                Tip: Add contacts in the <a href="/contacts" className="text-[#0071e3]">Contacts</a> tab to quickly select people here.
              </p>
            )}

            <button
              onClick={() => setStep('receipts')}
              disabled={people.length < 1}
              className="w-full bg-[#0071e3] text-white py-3 rounded-2xl text-sm font-semibold disabled:opacity-40 hover:bg-[#0077ed] transition-colors"
            >
              Continue with {people.length || 0} {people.length === 1 ? 'person' : 'people'} →
            </button>

            {/* Saved splits */}
            {(savedSplitsQuery.data?.length ?? 0) > 0 && (
              <div>
                <button
                  onClick={() => setShowSavedSplits(o => !o)}
                  className="w-full flex items-center justify-between text-xs text-[#6e6e73] py-2 hover:text-[#1d1d1f] transition-colors"
                >
                  <span className="font-medium">Saved splits ({savedSplitsQuery.data!.length})</span>
                  <span>{showSavedSplits ? '▲' : '▼'}</span>
                </button>

                {showSavedSplits && (
                  <div className="bg-white rounded-2xl border border-[#e8e8ed] overflow-hidden divide-y divide-[#f2f2f7]">
                    {savedSplitsQuery.data!.map(s => (
                      <div key={s.id} className="flex items-center justify-between px-4 py-3 hover:bg-[#f9f9f9] transition-colors">
                        <button
                          onClick={() => loadSplit(s)}
                          className="flex-1 text-left"
                        >
                          <p className="text-sm font-medium text-[#1d1d1f]">{s.title || 'Bill Split'}</p>
                          <p className="text-xs text-[#aeaeb2] mt-0.5">
                            {new Date(s.updated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                          </p>
                        </button>
                        <button
                          onClick={() => {
                            if (confirm('Delete this saved split?')) {
                              deleteSplit.mutate(s.id)
                              if (savedSplitId === s.id) setSavedSplitId(null)
                            }
                          }}
                          className={iconBtnDelete}
                          title="Delete"
                        >
                          <IconTrash />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── STEP 2: RECEIPTS ─────────────────────────────────────────────── */}
        {step === 'receipts' && (
          <div className="space-y-4">
            <div className="mb-2">
              <h2 className="text-2xl font-semibold text-[#1d1d1f] tracking-tight mb-1">Add receipts</h2>
              <p className="text-sm text-[#6e6e73]">Scan a receipt or enter items manually</p>
            </div>

            {/* Receipt cards (scanned + manual) */}
            {receipts.map(r => (
              <div key={r.id} className="bg-white rounded-2xl border border-[#e8e8ed] p-5">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1 min-w-0 mr-3">
                    <input
                      value={r.merchantName}
                      onChange={e => setReceipts(prev => prev.map(x => x.id === r.id ? { ...x, merchantName: e.target.value } : x))}
                      placeholder={r.isManual ? 'Bill name (e.g. Dinner)' : (r.scanning ? 'Scanning...' : 'Merchant')}
                      className="text-sm font-semibold text-[#1d1d1f] w-full bg-transparent border-0 focus:outline-none focus:bg-[#f5f5f7] rounded px-1 -ml-1 py-0.5"
                    />
                    {!r.isManual && r.items.length > 0 && (
                      <p className="text-xs font-medium mt-0.5">
                        {r.ocr_total
                          ? isTallied(r)
                            ? <span className="text-[#34c759]">✓ {r.items.length} items · matches ${r.ocr_total.toFixed(2)}</span>
                            : <span className="text-[#ff3b30]">⚠ {r.items.length} items · ${receiptItemsTotal(r).toFixed(2)} vs ${r.ocr_total.toFixed(2)}</span>
                          : <span className="text-[#34c759]">✓ {r.items.length} items found</span>
                        }
                      </p>
                    )}
                    {r.isManual && r.items.length > 0 && (
                      <p className="text-xs text-[#34c759] font-medium mt-0.5">✓ {r.items.length} item{r.items.length !== 1 ? 's' : ''} · ${receiptEffectiveTotal(r).toFixed(2)}</p>
                    )}
                    {r.error && <p className="text-xs text-[#ff3b30] mt-0.5">{r.error}</p>}
                  </div>
                  <button onClick={() => setReceipts(prev => prev.filter(x => x.id !== r.id))} className={`${iconBtnDelete} shrink-0`} title="Remove receipt">
                    <IconTrash />
                  </button>
                </div>

                {r.scanning ? (
                  <div className="flex items-center gap-2 text-sm text-[#6e6e73] py-2">
                    <div className="w-4 h-4 border-2 border-[#0071e3] border-t-transparent rounded-full animate-spin shrink-0" />
                    Reading receipt with AI...
                  </div>
                ) : r.isManual ? (
                  <div className="space-y-3">
                    {/* Existing items */}
                    {r.items.length > 0 && (
                      <div className="divide-y divide-[#f2f2f7] border border-[#f2f2f7] rounded-xl overflow-hidden">
                        {r.items.map(item => (
                          <div key={item.id} className="flex items-center justify-between px-3 py-2">
                            <span className="text-sm text-[#1d1d1f] flex-1 truncate">{item.name}</span>
                            <span className="text-sm font-medium text-[#1d1d1f] ml-3">${item.unitPrice.toFixed(2)}</span>
                            <button
                              onClick={() => setReceipts(prev => prev.map(rx => rx.id !== r.id ? rx : { ...rx, items: rx.items.filter(i => i.id !== item.id) }))}
                              className="ml-2 text-[#aeaeb2] hover:text-[#ff3b30] transition-colors text-xs"
                            >✕</button>
                          </div>
                        ))}
                      </div>
                    )}
                    {/* Add item row */}
                    <div className="flex gap-2">
                      <input
                        value={manualInputs[r.id]?.name || ''}
                        onChange={e => setManualInputs(prev => ({ ...prev, [r.id]: { ...prev[r.id], name: e.target.value } }))}
                        onKeyDown={e => { if (e.key === 'Enter') addManualItem(r.id) }}
                        placeholder="Item name"
                        className="flex-1 border border-[#d2d2d7] rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#0071e3] bg-[#f9f9f9]"
                      />
                      <div className="relative w-24">
                        <span className="absolute left-2.5 top-2.5 text-xs text-[#aeaeb2]">$</span>
                        <input
                          type="number" min="0" step="0.01"
                          value={manualInputs[r.id]?.price || ''}
                          onChange={e => setManualInputs(prev => ({ ...prev, [r.id]: { ...prev[r.id], price: e.target.value } }))}
                          onKeyDown={e => { if (e.key === 'Enter') addManualItem(r.id) }}
                          placeholder="0.00"
                          className="w-full border border-[#d2d2d7] rounded-xl pl-5 pr-2 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#0071e3] bg-[#f9f9f9]"
                        />
                      </div>
                      <button
                        onClick={() => addManualItem(r.id)}
                        disabled={!manualInputs[r.id]?.name?.trim() || !manualInputs[r.id]?.price}
                        className="bg-[#0071e3] text-white px-3 py-2 rounded-xl text-sm font-medium disabled:opacity-40 hover:bg-[#0077ed] transition-colors"
                      >Add</button>
                    </div>
                    {/* Paid by */}
                    <div>
                      <label className="text-xs text-[#86868b] mb-1.5 block">Paid by</label>
                      <select
                        value={r.paidById}
                        onChange={e => setReceipts(prev => prev.map(x => x.id === r.id ? { ...x, paidById: e.target.value } : x))}
                        className="w-full border border-[#d2d2d7] rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#0071e3] bg-[#f9f9f9]"
                      >
                        {people.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                      </select>
                    </div>
                  </div>
                ) : (
                  <div>
                    <label className="text-xs text-[#86868b] mb-1.5 block">Paid by</label>
                    <select
                      value={r.paidById}
                      onChange={e => setReceipts(prev => prev.map(x => x.id === r.id ? { ...x, paidById: e.target.value } : x))}
                      className="w-full border border-[#d2d2d7] rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#0071e3] bg-[#f9f9f9]"
                    >
                      {people.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                    </select>
                  </div>
                )}
              </div>
            ))}

            {/* Add options */}
            <div className="grid grid-cols-2 gap-3">
              {/* Scan receipt */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className="bg-white rounded-2xl border-2 border-dashed border-[#d2d2d7] p-6 text-center cursor-pointer hover:border-[#0071e3] hover:bg-[#fafcff] transition-colors"
              >
                <div className="w-9 h-9 bg-[#e8f1fb] rounded-xl flex items-center justify-center mx-auto mb-2">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0071e3" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="5" y="2" width="14" height="20" rx="2"/>
                    <line x1="9" y1="7" x2="15" y2="7"/>
                    <line x1="9" y1="11" x2="15" y2="11"/>
                    <line x1="9" y1="15" x2="13" y2="15"/>
                  </svg>
                </div>
                <p className="text-sm font-medium text-[#0071e3]">Scan receipts</p>
                <p className="text-xs text-[#86868b] mt-0.5">Select one or more photos</p>
              </div>

              {/* Enter manually */}
              <div
                onClick={addManualReceipt}
                className="bg-white rounded-2xl border-2 border-dashed border-[#d2d2d7] p-6 text-center cursor-pointer hover:border-[#34c759] hover:bg-[#f0fdf4] transition-colors"
              >
                <div className="w-9 h-9 bg-[#e8faf0] rounded-xl flex items-center justify-center mx-auto mb-2">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#34c759" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="5" x2="12" y2="19"/>
                    <line x1="5" y1="12" x2="19" y2="12"/>
                  </svg>
                </div>
                <p className="text-sm font-medium text-[#34c759]">Enter manually</p>
                <p className="text-xs text-[#86868b] mt-0.5">Type items & prices</p>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,.heic,.heif,.pdf,application/pdf"
              multiple
              className="hidden"
              onChange={e => {
                const files = Array.from(e.target.files || [])
                if (!files.length) return
                handleFiles(files)
                e.target.value = ''
              }}
            />

            <div className="flex gap-3 pt-2">
              <button onClick={() => setStep('people')} className="px-4 py-2.5 rounded-xl text-sm text-[#6e6e73] hover:bg-[#f2f2f7] transition-colors">← Back</button>
              <button
                onClick={() => setStep('review')}
                disabled={receipts.length === 0 || receipts.some(r => r.scanning) || receipts.some(r => r.items.length === 0)}
                className="flex-1 bg-[#0071e3] text-white py-2.5 rounded-2xl text-sm font-semibold disabled:opacity-40 hover:bg-[#0077ed] transition-colors"
              >
                Assign items →
              </button>
            </div>
          </div>
        )}

        {/* ── STEP 3: REVIEW ───────────────────────────────────────────────── */}
        {step === 'review' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-semibold text-[#1d1d1f] tracking-tight mb-1">Assign items</h2>
                <p className="text-sm text-[#6e6e73]">
                  {receipts.length > 1
                    ? `${receipts.length} receipts · tap names to assign, tap price to edit`
                    : 'Tap names to assign. Tap a price to edit it.'}
                </p>
              </div>
              <button onClick={() => setStep('summary')} className="bg-[#0071e3] text-white px-4 py-2 rounded-xl text-sm font-semibold hover:bg-[#0077ed] transition-colors">
                Summary →
              </button>
            </div>

            {/* Receipt jump pills (shown when 2+ receipts) */}
            {receipts.length > 1 && (
              <div className="flex flex-wrap gap-2">
                {receipts.map((r, i) => (
                  <a
                    key={r.id}
                    href={`#receipt-${r.id}`}
                    className="text-xs px-3 py-1.5 rounded-full bg-white border border-[#e8e8ed] text-[#3a3a3c] hover:border-[#0071e3] hover:text-[#0071e3] transition-colors font-medium"
                  >
                    {r.merchantName || `Receipt ${i + 1}`}
                    {r.items.filter(x => !x.isTaxLine && !x.isTipLine).length > 0
                      ? ` · ${r.items.filter(x => !x.isTaxLine && !x.isTipLine).length} items`
                      : ''}
                  </a>
                ))}
              </div>
            )}

            {receipts.map(r => (
              <div key={r.id} id={`receipt-${r.id}`} className="bg-white rounded-2xl border border-[#e8e8ed] overflow-hidden scroll-mt-4">
                {/* Receipt header */}
                <div className="px-5 py-4 bg-[#f9f9f9] border-b border-[#e8e8ed]">
                  <div className="flex items-start justify-between mb-3 gap-4">
                    <div className="flex-1 min-w-0">
                      {/* Editable name */}
                      <input
                        value={r.merchantName}
                        onChange={e => setReceipts(prev => prev.map(x => x.id === r.id ? { ...x, merchantName: e.target.value } : x))}
                        placeholder="Receipt name"
                        className="text-sm font-semibold text-[#1d1d1f] w-full bg-transparent border-0 focus:outline-none focus:bg-white focus:ring-1 focus:ring-[#0071e3] rounded-lg px-1 -ml-1 py-0.5 transition-all"
                      />
                      {/* Paid by + Date row */}
                      <div className="flex items-center gap-3 mt-1 flex-wrap">
                        <select
                          value={r.paidById}
                          onChange={e => setReceipts(prev => prev.map(x => x.id === r.id ? { ...x, paidById: e.target.value } : x))}
                          className="text-xs text-[#86868b] bg-transparent border-0 focus:outline-none focus:ring-1 focus:ring-[#0071e3] rounded cursor-pointer -ml-0.5"
                        >
                          {people.map(p => <option key={p.id} value={p.id}>Paid by {p.name}</option>)}
                        </select>
                        <span className="text-[#d2d2d7] text-xs">·</span>
                        <input
                          type="date"
                          value={r.date || ''}
                          onChange={e => setReceipts(prev => prev.map(x => x.id === r.id ? { ...x, date: e.target.value } : x))}
                          className="text-xs text-[#86868b] bg-transparent border-0 focus:outline-none focus:ring-1 focus:ring-[#0071e3] rounded cursor-pointer"
                        />
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <p className="text-base font-bold text-[#1d1d1f]">${receiptEffectiveTotal(r).toFixed(2)}</p>
                      {r.ocr_total && (
                        <p className="text-xs mt-0.5">
                          {isTallied(r)
                            ? <span className="text-[#34c759] font-medium">✓ matches receipt total</span>
                            : <span className="text-[#ff3b30] font-medium">off by ${Math.abs(receiptEffectiveTotal(r) - r.ocr_total).toFixed(2)}</span>
                          }
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Tally row */}
                  {(() => {
                    const tax = effectiveTaxTotal(r) + embeddedTaxTotal(r)
                    const tip = tipTotal(r)
                    const pretax = receiptEffectiveTotal(r) - tax - tip
                    return (
                      <div className="flex items-center gap-2 text-xs mt-1 flex-wrap">
                        <span className="text-[#86868b]">Subtotal <strong className="text-[#1d1d1f]">${pretax.toFixed(2)}</strong></span>
                        {tax > 0.01 && <>
                          <span className="text-[#aeaeb2]">+</span>
                          <span className="text-[#86868b]">Tax <strong className="text-[#1d1d1f]">${tax.toFixed(2)}</strong></span>
                        </>}
                        {tip > 0.01 && <>
                          <span className="text-[#aeaeb2]">+</span>
                          <span className="text-[#86868b]">Tip <strong className="text-[#1d1d1f]">${tip.toFixed(2)}</strong></span>
                        </>}
                        {r.ocr_total && (
                          isTallied(r)
                            ? <span className="text-[#34c759] font-semibold ml-auto">✓</span>
                            : <span className="text-[#ff3b30] font-semibold ml-auto">⚠ off by ${Math.abs(receiptEffectiveTotal(r) - r.ocr_total).toFixed(2)}</span>
                        )}
                      </div>
                    )
                  })()}
                </div>

                {r.items.map(item => {
                  const effective = itemTotal(item)
                  const base = item.unitPrice * item.quantity
                  const discounted = Math.max(0, base - (item.discountAmount || 0))
                  const taxAmt = item.isTaxable ? discounted * (item.taxRate / 100) : 0

                  // Tax / tip lines — compact read-only row, auto-split equally
                  if (item.isTaxLine || item.isTipLine) {
                    const effectiveAmt = item.isTaxLine ? effectiveTaxLineAmount(r, item) : itemTotal(item)
                    const label = item.isTipLine ? 'Tip' : 'Tax'
                    const color = item.isTipLine ? 'text-[#1a6080] bg-[#e8f1fb]' : 'text-[#b35a00] bg-[#fff5e6]'
                    const perPerson = people.length > 0 ? effectiveAmt / people.length : 0
                    return (
                      <div key={item.id} className="border-b border-[#f2f2f7] last:border-0 px-5 py-3 flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${color}`}>{label}</span>
                          <span className="text-sm text-[#86868b]">{item.name}</span>
                        </div>
                        <div className="flex items-center gap-3 shrink-0">
                          {effectiveAmt > 0.01 && perPerson > 0 && (
                            <span className="text-xs text-[#aeaeb2]">${perPerson.toFixed(2)}/person</span>
                          )}
                          <span className="text-sm font-semibold text-[#1d1d1f]">${effectiveAmt.toFixed(2)}</span>
                          <span className="text-xs text-[#34c759] font-medium">✓ auto</span>
                        </div>
                      </div>
                    )
                  }

                  const isEditing = editingItemId === item.id
                  return (
                    <div key={item.id} className="border-b border-[#f2f2f7] last:border-0 px-5 py-4">
                      {/* Read-only row: name left, price + actions right */}
                      <div className="flex items-start justify-between mb-2.5">
                        <span className="text-sm font-medium text-[#1d1d1f] flex-1 mr-3 leading-snug">{item.name}</span>
                        <div className="flex items-center gap-1 shrink-0">
                          <div className="text-right mr-1">
                            <span className={`text-sm font-semibold ${isEditing ? 'text-[#0071e3]' : 'text-[#1d1d1f]'}`}>
                              ${effective.toFixed(2)}
                            </span>
                            {item.isTaxable && taxAmt > 0.001 && (
                              <p className="text-xs text-[#b35a00] mt-0.5">${discounted.toFixed(2)} +${taxAmt.toFixed(2)} tax</p>
                            )}
                            {item.discountAmount > 0 && (
                              <p className="text-xs text-[#34c759] mt-0.5">-${item.discountAmount.toFixed(2)} discount</p>
                            )}
                          </div>
                          <button
                            onClick={() => setEditingItemId(isEditing ? null : item.id)}
                            className={iconBtnEdit}
                            title="Edit item"
                          >
                            <IconPencil />
                          </button>
                          <button
                            onClick={() => {
                              setReceipts(prev => prev.map(rx => rx.id !== r.id ? rx : { ...rx, items: rx.items.filter(i => i.id !== item.id) }))
                              if (editingItemId === item.id) setEditingItemId(null)
                            }}
                            className={iconBtnDelete}
                            title="Delete item"
                          >
                            <IconTrash />
                          </button>
                        </div>
                      </div>

                      {/* Inline edit panel (shown on tap) */}
                      {isEditing && (
                        <div className="bg-[#f9f9f9] rounded-xl p-3 mb-3 space-y-3 border border-[#e8e8ed]">
                          {/* Name */}
                          <div>
                            <p className="text-xs text-[#86868b] mb-1">Name</p>
                            <input
                              value={item.name}
                              onChange={e => updateItem(r.id, item.id, { name: e.target.value })}
                              className="w-full border border-[#d2d2d7] rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-[#0071e3] bg-white"
                            />
                          </div>
                          {/* Qty × Price | Discount | Tax */}
                          <div className="grid grid-cols-3 gap-2.5">
                            <div>
                              <p className="text-xs text-[#86868b] mb-1">Qty × Price</p>
                              <div className="flex items-center gap-1">
                                <input
                                  type="number" min="1" step="1"
                                  value={item.quantity}
                                  onChange={e => updateItem(r.id, item.id, { quantity: parseFloat(e.target.value) || 1 })}
                                  className="w-9 border border-[#d2d2d7] rounded-lg px-1 py-1 text-xs text-center focus:outline-none focus:ring-1 focus:ring-[#0071e3] bg-white"
                                />
                                <span className="text-[#aeaeb2] text-xs">×</span>
                                <div className="relative flex-1">
                                  <span className="absolute left-1.5 top-1 text-xs text-[#aeaeb2]">$</span>
                                  <input
                                    type="number" min="0" step="0.01"
                                    value={item.unitPrice}
                                    onChange={e => updateItem(r.id, item.id, { unitPrice: parseFloat(e.target.value) || 0 })}
                                    className="w-full border border-[#d2d2d7] rounded-lg pl-4 pr-1 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-[#0071e3] bg-white"
                                  />
                                </div>
                              </div>
                            </div>
                            <div>
                              <p className="text-xs text-[#86868b] mb-1">Discount</p>
                              <div className="relative">
                                <span className="absolute left-2 top-1 text-xs text-[#aeaeb2]">-$</span>
                                <input
                                  type="number" min="0" step="0.01"
                                  value={item.discountAmount || ''}
                                  placeholder="0.00"
                                  onChange={e => updateItem(r.id, item.id, { discountAmount: parseFloat(e.target.value) || 0 })}
                                  className="w-full border border-[#d2d2d7] rounded-lg pl-5 pr-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-[#0071e3] bg-white"
                                />
                              </div>
                            </div>
                            <div>
                              <label className="flex items-center gap-1.5 text-xs text-[#86868b] mb-1 cursor-pointer select-none">
                                <input
                                  type="checkbox"
                                  checked={item.isTaxable}
                                  onChange={e => updateItem(r.id, item.id, { isTaxable: e.target.checked })}
                                  className="w-3 h-3 accent-[#0071e3]"
                                />
                                Tax
                              </label>
                              <div className="relative">
                                <input
                                  type="number" min="0" max="100" step="0.01"
                                  value={item.taxRate}
                                  disabled={!item.isTaxable}
                                  onChange={e => updateItem(r.id, item.id, { taxRate: parseFloat(e.target.value) || 0 })}
                                  className="w-full border border-[#d2d2d7] rounded-lg pl-2 pr-5 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-[#0071e3] disabled:opacity-30 disabled:bg-[#f2f2f7] bg-white"
                                />
                                <span className="absolute right-2 top-1 text-xs text-[#aeaeb2]">%</span>
                              </div>
                            </div>
                          </div>
                          <div className="flex justify-end">
                            <button
                              onClick={() => setEditingItemId(null)}
                              className="text-xs bg-[#0071e3] text-white px-3 py-1.5 rounded-lg font-medium hover:bg-[#0077ed] transition-colors"
                            >
                              Done
                            </button>
                          </div>
                        </div>
                      )}

                      {/* Assign people */}
                      <div className="flex flex-wrap gap-1.5">
                        <button
                          onClick={() => {
                            const allIn = people.every(p => item.assignedTo.includes(p.id))
                            setReceipts(prev => prev.map(rx => rx.id !== r.id ? rx : {
                              ...rx,
                              items: rx.items.map(i => i.id !== item.id ? i : {
                                ...i, assignedTo: allIn ? [] : people.map(p => p.id)
                              })
                            }))
                          }}
                          className="px-2.5 py-1 rounded-full text-xs font-medium bg-[#f2f2f7] text-[#6e6e73] hover:bg-[#e8e8ed] transition-colors"
                        >
                          {people.every(p => item.assignedTo.includes(p.id)) ? 'Clear' : 'All'}
                        </button>
                        {people.map(person => {
                          const on = item.assignedTo.includes(person.id)
                          const share = on && item.assignedTo.length > 0 ? effective / item.assignedTo.length : null
                          return (
                            <button
                              key={person.id}
                              onClick={() => togglePerson(r.id, item.id, person.id)}
                              className="px-3 py-1 rounded-full text-xs font-medium transition-colors"
                              style={{ background: on ? avatarColor(person.name).bg : '#f2f2f7', color: on ? avatarColor(person.name).text : '#3a3a3c' }}
                            >
                              {person.name}{share !== null ? ` $${share.toFixed(2)}` : ''}
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  )
                })}
              </div>
            ))}

            {(() => {
              const untallied = receipts.filter(r => !isTallied(r))
              const taxUnassigned = false // tax lines are auto-assigned on scan
              const doubleTax = receipts.some(r => hasDoubleTax(r))
              const canProceed = untallied.length === 0 && !taxUnassigned && !doubleTax
              return (
                <div className="space-y-2 pt-2">
                  {untallied.length > 0 && (
                    <div className="bg-[#fff2f2] border border-[#ffd0d0] rounded-2xl px-4 py-3 text-xs text-[#ff3b30] font-medium">
                      ⚠️ {untallied.length} receipt{untallied.length > 1 ? 's' : ''} {untallied.length > 1 ? 'have' : 'has'} item totals that don't match the scanned total. Fix before proceeding.
                    </div>
                  )}
                  {taxUnassigned && (
                    <div className="bg-[#fff5e6] border border-[#ffd9a8] rounded-2xl px-4 py-3 text-xs text-[#b35a00] font-medium">
                      ⚠️ Some tax lines haven't been assigned to anyone yet.
                    </div>
                  )}
                  {doubleTax && (
                    <div className="bg-[#fff8ec] border border-[#ffd070] rounded-2xl px-4 py-3 text-xs text-[#7d4e00] font-medium">
                      ⚠️ Tax is being double-counted on one or more receipts. Uncheck the Tax checkbox on individual items where a TAX line already exists.
                    </div>
                  )}
                  <div className="flex gap-3">
                    <button onClick={() => setStep('receipts')} className="px-4 py-2.5 rounded-xl text-sm text-[#6e6e73] hover:bg-[#f2f2f7] transition-colors">← Back</button>
                    <button
                      onClick={() => setStep('summary')}
                      disabled={!canProceed}
                      title={!canProceed ? 'Fix tally issues above first' : ''}
                      className="flex-1 bg-[#0071e3] text-white py-2.5 rounded-2xl text-sm font-semibold hover:bg-[#0077ed] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      See who owes what →
                    </button>
                  </div>
                </div>
              )
            })()}
          </div>
        )}

        {/* ── STEP 4: SUMMARY ──────────────────────────────────────────────── */}
        {step === 'summary' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-2xl font-semibold text-[#1d1d1f] tracking-tight">Summary</h2>
              <button onClick={() => setStep('review')} className="text-sm text-[#0071e3] hover:text-[#0077ed]">← Edit</button>
            </div>

            {/* Grand total */}
            <div className="bg-white rounded-2xl border border-[#e8e8ed] px-5 py-4 flex items-center justify-between">
              <p className="text-sm font-medium text-[#6e6e73]">Grand total</p>
              <p className="text-2xl font-bold text-[#1d1d1f]">${grandTotal.toFixed(2)}</p>
            </div>

            {/* Per-person */}
            <div className="bg-white rounded-2xl border border-[#e8e8ed] divide-y divide-[#f2f2f7]">
              {summary.map(({ person, share, paid, net }) => (
                <div key={person.id} className="px-5 py-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-full text-white flex items-center justify-center text-sm font-semibold" style={{ background: avatarColor(person.name).bg, color: avatarColor(person.name).text }}>
                      {person.name[0].toUpperCase()}
                    </div>
                    <div>
                      <p className="text-sm font-medium text-[#1d1d1f]">{person.name}</p>
                      {paid > 0 && <p className="text-xs text-[#86868b]">paid ${paid.toFixed(2)}</p>}
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-semibold text-[#1d1d1f]">share ${share.toFixed(2)}</p>
                    {net > 0.01 && <p className="text-xs font-semibold text-[#ff9500]">owes ${net.toFixed(2)}</p>}
                    {net < -0.01 && <p className="text-xs font-semibold text-[#34c759]">gets back ${Math.abs(net).toFixed(2)}</p>}
                    {Math.abs(net) <= 0.01 && <p className="text-xs text-[#aeaeb2]">settled</p>}
                  </div>
                </div>
              ))}
            </div>

            {/* Settlements */}
            {settlements.length > 0 && (
              <div className="bg-white rounded-2xl border border-[#e8e8ed] p-5">
                <p className="text-xs font-semibold text-[#86868b] uppercase tracking-widest mb-4">Who pays who</p>
                <div className="space-y-3">
                  {settlements.map((s, i) => (
                    <div key={i} className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-sm">
                        <span className="font-medium text-[#1d1d1f]">{s.from}</span>
                        <svg width="14" height="10" viewBox="0 0 14 10" fill="none">
                          <path d="M1 5h12M9 1l4 4-4 4" stroke="#aeaeb2" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        <span className="font-medium text-[#1d1d1f]">{s.to}</span>
                      </div>
                      <span className="text-sm font-bold text-[#0071e3]">${s.amount.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Receipt breakdown */}
            <div className="bg-white rounded-2xl border border-[#e8e8ed] p-5">
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs font-semibold text-[#86868b] uppercase tracking-widest">
                  {receipts.length > 1 ? `${receipts.length} Receipts` : 'Receipt'}
                </p>
                <button
                  onClick={() => setStep('review')}
                  className="text-xs text-[#0071e3] hover:text-[#0077ed] font-medium transition-colors"
                >
                  Edit all
                </button>
              </div>
              {receipts.map((r, i) => {
                const regItems = r.items.filter(x => !x.isTaxLine && !x.isTipLine)
                const tipAmt = r.items.filter(x => x.isTipLine).reduce((s, x) => s + itemTotal(x), 0)
                const taxAmt = embeddedTaxTotal(r) + effectiveTaxTotal(r)
                return (
                  <div key={r.id} className="py-2.5 border-b border-[#f2f2f7] last:border-0">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 min-w-0 flex-1">
                        <span className="text-sm font-medium text-[#1d1d1f] truncate">{r.merchantName || `Receipt ${i + 1}`}</span>
                        <span className="text-xs text-[#aeaeb2] shrink-0">{regItems.length} item{regItems.length !== 1 ? 's' : ''}</span>
                      </div>
                      <div className="flex items-center gap-3 shrink-0 ml-3">
                        <span className="text-sm font-semibold text-[#1d1d1f]">${receiptEffectiveTotal(r).toFixed(2)}</span>
                        <button
                          onClick={() => {
                            setStep('review')
                            // Brief delay so review step renders before scrolling
                            setTimeout(() => {
                              document.getElementById(`receipt-${r.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                            }, 80)
                          }}
                          className="text-xs text-[#0071e3] hover:text-[#0077ed] font-medium transition-colors"
                        >
                          Edit
                        </button>
                      </div>
                    </div>
                    {/* Subtotals row */}
                    <div className="flex gap-3 mt-0.5 text-xs text-[#86868b]">
                      <span>Subtotal ${(receiptEffectiveTotal(r) - taxAmt - tipAmt).toFixed(2)}</span>
                      {taxAmt > 0.01 && <span>Tax ${taxAmt.toFixed(2)}</span>}
                      {tipAmt > 0.01 && <span>Tip ${tipAmt.toFixed(2)}</span>}
                    </div>
                    {/* Paid by */}
                    {(() => {
                      const payer = people.find(p => p.id === r.paidById)
                      return payer ? <p className="text-xs text-[#aeaeb2] mt-0.5">Paid by {payer.name}</p> : null
                    })()}
                  </div>
                )
              })}
            </div>

            {/* Save split */}
            {saveSplit.isSuccess && !saveSplit.isPending ? (
              <div className="bg-[#f0fdf4] border border-[#bbf7d0] rounded-2xl px-5 py-4 flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold text-[#15803d]">✓ Split saved</p>
                  <p className="text-xs text-[#86868b] mt-0.5">You can reload it anytime from the home screen</p>
                </div>
                <button
                  onClick={() => saveSplit.mutate()}
                  className="text-xs text-[#0071e3] font-medium hover:underline"
                >
                  Update
                </button>
              </div>
            ) : (
              <button
                onClick={() => saveSplit.mutate()}
                disabled={saveSplit.isPending}
                className="w-full bg-[#1d1d1f] text-white py-3 rounded-2xl text-sm font-semibold hover:bg-[#3a3a3c] transition-colors disabled:opacity-50"
              >
                {saveSplit.isPending ? 'Saving…' : savedSplitId ? 'Update saved split' : 'Save split'}
              </button>
            )}
            {saveSplit.isError && (
              <p className="text-xs text-[#ff3b30] text-center">Failed to save — try again</p>
            )}

            <button
              onClick={() => { setPeople([]); setReceipts([]); setSavedSplitId(null); saveSplit.reset(); setStep('people') }}
              className="w-full text-sm text-[#86868b] hover:text-[#1d1d1f] py-3 rounded-2xl hover:bg-[#f2f2f7] transition-colors"
            >
              Start a new split
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
